import os
import math
import torch
import torch.nn as nn
import sys
import csv

# Ensure SentencePiece is installed.
try:
    import sentencepiece
except ImportError:
    sys.exit("SentencePiece is required. Please run: pip install sentencepiece and restart your runtime.")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# ------------------------------------------------------------------
# 1) Reading Q&A Data from medquad.csv
# ------------------------------------------------------------------
def parse_medquad_data(file_path="medquad.csv"):
    """
    Reads medquad.csv, which has two columns: question, answer.
    Merges each row into "Question: <question>\nAnswer: <answer>"
    Returns a list of these merged Q&A strings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    merged_samples = []
    import csv
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # Assumes headers: question, answer
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            # Merge them into a single text
            text = f"Question: {q}\nAnswer: {a}"
            merged_samples.append(text)

    return merged_samples

def create_qa_dataset(qa_samples):
    return Dataset.from_dict({"text": qa_samples})

# ------------------------------------------------------------------
# 2) Rank-1 LoRA with Dynamic Pruning (with α clamping)
# ------------------------------------------------------------------
class RankOneLoRAWithGating(nn.Module):
    """
    Implements LoRA as the sum of rank-1 outer products:
      ΔW = Σ_i [α_i * a_i (b_i)^T].
    The base layer is frozen; dimensions are derived from its weight.
    For OpenLLaMA's q_proj, we assume base_layer.weight.shape is [in_dim, out_dim].
    """
    def __init__(self, base_layer, rank=8, init_alpha=0.1):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Assume base_layer.weight shape is [in_dim, out_dim]
        self.in_dim, self.out_dim = base_layer.weight.shape
        dtype = base_layer.weight.dtype
        self.rank = rank

        # Create LoRA parameters
        self.a_vectors = nn.Parameter(torch.randn(rank, self.in_dim, dtype=dtype) * 0.01)
        self.b_vectors = nn.Parameter(torch.randn(rank, self.out_dim, dtype=dtype) * 0.01)
        self.alpha = nn.Parameter(torch.ones(rank, dtype=dtype) * init_alpha)
        self.scale = 1.0 / math.sqrt(rank)

    def forward(self, x: torch.Tensor):
        # x: [B, T, in_dim]
        B, T, _ = x.shape
        x_2d = x.view(B * T, self.in_dim)

        # If base_layer has no bias, create a zero bias
        bias = self.base_layer.bias if self.base_layer.bias is not None else torch.zeros(self.out_dim, device=x.device, dtype=x.dtype)
        base_2d = torch.addmm(bias, x_2d, self.base_layer.weight)

        # LoRA update
        s = torch.matmul(x_2d, self.a_vectors.transpose(0,1))  # [B*T, rank]
        s_3d = s.unsqueeze(-1)                                 # [B*T, rank, 1]
        b_3d = self.b_vectors.unsqueeze(0)                     # [1, rank, out_dim]

        alpha_clamped = torch.clamp(self.alpha, min=1e-7, max=1e2)
        alpha_3d = alpha_clamped.unsqueeze(-1).unsqueeze(0)    # [1, rank, 1]

        lora_2d = (s_3d * b_3d * alpha_3d * self.scale).sum(dim=1)  # [B*T, out_dim]
        out_2d = base_2d + lora_2d
        return out_2d.view(B, T, self.out_dim)

# ------------------------------------------------------------------
# 3) OpenLLaMA Model with Dynamic LoRA Injection + generate() Method
# ------------------------------------------------------------------
class LoRAOpenLLaMA(nn.Module):
    """
    Loads OpenLLaMA and replaces the first block's self_attn.q_proj with our custom dynamic LoRA layer.
    Exposes a generate() method for text generation.
    """
    def __init__(self, base_model_name="openlm-research/open_llama_3b", rank=8):
        super().__init__()
        print(f"Loading OpenLLaMA from {base_model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            self.base_model = self.base_model.to("cuda")

        # Replace q_proj in the first transformer block
        first_block = self.base_model.model.layers[0]
        old_q_proj = first_block.self_attn.q_proj
        first_block.self_attn.q_proj = RankOneLoRAWithGating(base_layer=old_q_proj, rank=rank, init_alpha=0.1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)

# ------------------------------------------------------------------
# 4) Tokenization & Data Collation
# ------------------------------------------------------------------
def tokenize_fn(examples, tokenizer, max_len=128):
    return tokenizer(examples["text"], max_length=max_len, truncation=True, padding="max_length")

def data_collator(examples, pad_id):
    input_ids = torch.stack([ex["input_ids"] for ex in examples])
    attention_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

# ------------------------------------------------------------------
# 5) Main: Fine-Tune LoRA on medquad.csv and Compare with Baseline OpenLLaMA
# ------------------------------------------------------------------
def main():
    # A) Load Q&A data from medquad.csv
    from transformers import Trainer, TrainerCallback

    file_path = r"C:\backupcgi\000Ed_CGI\medquad.csv"
    qa_samples = parse_medquad_data(file_path)
    if not qa_samples:
        raise ValueError("No Q&A found in medquad.csv.")
    print(f"Loaded {len(qa_samples)} Q&A lines from {file_path}.")
    dataset = create_qa_dataset(qa_samples)

    # B) Load baseline OpenLLaMA (no LoRA)
    baseline_model_name = "openlm-research/open_llama_3b"
    print("Loading baseline OpenLLaMA (no LoRA)...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    baseline_model = AutoModelForCausalLM.from_pretrained(
        baseline_model_name,
        torch_dtype=torch.float16
    )
    if torch.cuda.is_available():
        baseline_model = baseline_model.to("cuda")

    # C) Create LoRA-augmented OpenLLaMA
    print("Creating LoRA-augmented OpenLLaMA (rank=8) overriding first q_proj...")
    lora_model = LoRAOpenLLaMA(base_model_name=baseline_model_name, rank=8)

    # Freeze all parameters except LoRA's (a_vectors, b_vectors, alpha)
    for name, param in lora_model.named_parameters():
        if any(x in name for x in ["a_vectors", "b_vectors", "alpha"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # D) Tokenize the dataset
    tokenized_dataset = dataset.map(lambda ex: tokenize_fn(ex, tokenizer, max_len=128), batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    def my_data_collator(batch):
        return data_collator(batch, tokenizer.pad_token_id)

    # E) Training Arguments
    training_args = TrainingArguments(
        output_dir="./lora_openllama_medquad",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="no",  # disable checkpoint saving
        evaluation_strategy="no",
        do_train=True,
        do_eval=False,
        save_safetensors=False
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=my_data_collator
    )

    # F) Dynamic Alpha Pruning Callback
    alpha_threshold = 0.02
    patience_steps = 10
    alpha_history = {}

    class AlphaPruningCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # Check alpha in q_proj
            if state.global_step % 5 == 0 and state.global_step > 0:
                q_proj_lora = lora_model.base_model.model.layers[0].self_attn.q_proj
                alpha = q_proj_lora.alpha.detach().cpu()
                for i, val in enumerate(alpha):
                    if i not in alpha_history:
                        alpha_history[i] = []
                    alpha_history[i].append(float(val))
                # If alpha_i remains below threshold for patience_steps, prune it
                for i in range(len(alpha)):
                    if len(alpha_history[i]) >= patience_steps:
                        recent_vals = alpha_history[i][-patience_steps:]
                        if all(a_ < alpha_threshold for a_ in recent_vals):
                            with torch.no_grad():
                                q_proj_lora.alpha[i] = 0.0
                            print(f"[Prune] alpha[{i}] pruned at step {state.global_step}")

    trainer.add_callback(AlphaPruningCallback())

    # G) Fine-Tune the LoRA Model
    print("Training the LoRA-augmented OpenLLaMA on medquad.csv Q&A data...")
    trainer.train()
    print("Training complete.\n")

    # H) Print final alpha values
    q_proj_layer = lora_model.base_model.model.layers[0].self_attn.q_proj
    final_alpha = q_proj_layer.alpha.detach().cpu().numpy()
    print("Final alpha array:", final_alpha)
    active_count = sum(a != 0 for a in final_alpha)
    print(f"Active directions: {active_count}/{len(final_alpha)}\n")

    # I) Compare: Baseline vs. LoRA
    def generate_response(model, prompt):
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    test_question = "What causes Glaucoma ?"
    print("--- Baseline OpenLLaMA Response ---")
    print(generate_response(baseline_model, test_question))
    print("--- LoRA-Fine-Tuned OpenLLaMA Response ---")
    print(generate_response(lora_model, test_question))

if __name__ == "__main__":
    main()

