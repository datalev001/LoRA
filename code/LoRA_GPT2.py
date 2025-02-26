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
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# ------------------------------------------------------------------
# 1) Reading Data from medquad.csv
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
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # Assumes headers: question, answer
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            # Merge them into a single text for training
            text = f"Question: {q}\nAnswer: {a}"
            merged_samples.append(text)

    return merged_samples

def create_dataset(samples):
    return Dataset.from_dict({"text": samples})

# ------------------------------------------------------------------
# 2) Dynamic LoRA c_attn Module with Correct Dimension Handling
# ------------------------------------------------------------------
class DynamicLoRACAttn(nn.Module):
    """
    Replaces GPT-2's c_attn (a Conv1D layer). This module stores the base weight and bias,
    computes the base transform, and adds LoRA updates defined as:
      ΔW = Σ_i [α_i * a_i (b_i)^T].
    It automatically detects whether to transpose the base weight.
    """
    def __init__(self, base_weight, base_bias, rank=8, init_alpha=0.1):
        super().__init__()
        self.register_buffer("base_weight", base_weight)  # shape: [out_dim, in_dim]
        self.register_buffer("base_bias", base_bias)      # shape: [out_dim]
        self.rank = rank

        self.weight_shape = self.base_weight.shape  # e.g., (2304, 768)
        if self.weight_shape[0] > self.weight_shape[1]:
            self.out_dim, self.in_dim = self.weight_shape
            self.transpose_needed = True
        else:
            self.in_dim, self.out_dim = self.weight_shape
            self.transpose_needed = False

        print(f"Base c_attn weight shape: {self.weight_shape} -> in_dim={self.in_dim}, out_dim={self.out_dim}, transpose={self.transpose_needed}")

        self.a_vectors = nn.Parameter(torch.randn(rank, self.in_dim) * 0.01)
        self.b_vectors = nn.Parameter(torch.randn(rank, self.out_dim) * 0.01)
        self.alpha = nn.Parameter(torch.ones(rank) * init_alpha)
        self.scale = 1.0 / math.sqrt(rank)

    def forward(self, x: torch.Tensor):
        # x: [B, T, in_dim]
        B, T, C = x.shape
        x_2d = x.view(B * T, C)
        if self.transpose_needed:
            base_2d = torch.addmm(self.base_bias, x_2d, self.base_weight.transpose(0,1))
        else:
            base_2d = torch.addmm(self.base_bias, x_2d, self.base_weight)
        s = torch.matmul(x_2d, self.a_vectors.transpose(0,1))  # [B*T, rank]
        s_3d = s.unsqueeze(-1)  # [B*T, rank, 1]
        b_3d = self.b_vectors.unsqueeze(0)  # [1, rank, out_dim]
        alpha_3d = self.alpha.unsqueeze(-1).unsqueeze(0)  # [1, rank, 1]
        lora_2d = (s_3d * b_3d * alpha_3d * self.scale).sum(dim=1)  # [B*T, out_dim]
        out_2d = base_2d + lora_2d
        return out_2d.view(B, T, self.out_dim)

# ------------------------------------------------------------------
# 3) GPT-2 Model with Dynamic LoRA c_attn Injection and generate() Method
# ------------------------------------------------------------------
class NovelLoRAGPT2(nn.Module):
    """
    Loads GPT-2 and replaces the first block's attn.c_attn with our custom dynamic LoRA module.
    Exposes a generate() method for text generation.
    """
    def __init__(self, base_model_name="gpt2", rank=8):
        super().__init__()
        print(f"Loading GPT-2 from {base_model_name}...")
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name, torch_dtype=torch.float32)
        if torch.cuda.is_available():
            self.base_model = self.base_model.to("cuda")
        block0_attn = self.base_model.transformer.h[0].attn
        old_c_attn = block0_attn.c_attn
        base_weight = old_c_attn.weight.detach().clone()
        base_bias = old_c_attn.bias.detach().clone()
        new_c_attn = DynamicLoRACAttn(base_weight=base_weight, base_bias=base_bias, rank=rank, init_alpha=0.1)
        block0_attn.c_attn = new_c_attn

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def generate(self, **kwargs):
        # Clamp α before generation for stability.
        self.base_model.transformer.h[0].attn.c_attn.alpha.data.clamp_(min=1e-7, max=1e2)
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
# 5) Main: Fine-Tune LoRA on medquad.csv and Compare with Baseline GPT-2
# ------------------------------------------------------------------
def main():
    # A) Load data from medquad.csv
    file_path = r"C:\backupcgi\000Ed_CGI\medquad.csv"
    samples = parse_medquad_data(file_path)
    if not samples:
        raise ValueError("No data found in medquad.csv.")
    print(f"Loaded {len(samples)} Q&A samples from {file_path}.")
    dataset = create_dataset(samples)
    
    # B) Load baseline GPT-2 (no LoRA)
    baseline_model_name = "gpt2"
    print("Loading baseline GPT-2 for comparison...")
    baseline_model = GPT2LMHeadModel.from_pretrained(baseline_model_name, torch_dtype=torch.float32)
    tokenizer = GPT2Tokenizer.from_pretrained(baseline_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        baseline_model = baseline_model.to("cuda")
    
    # C) Create LoRA-augmented GPT-2
    print("Creating LoRA-augmented GPT-2 (rank=8) overriding c_attn forward...")
    lora_model = NovelLoRAGPT2(base_model_name=baseline_model_name, rank=8)
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
    from transformers import Trainer, TrainerCallback
    training_args = TrainingArguments(
        output_dir="./lora_gpt2_medquad",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="no",  # disable checkpoint saving to avoid file write errors
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
            c_attn_lora = lora_model.base_model.transformer.h[0].attn.c_attn
            alpha = c_attn_lora.alpha.detach().cpu()
            for i, val in enumerate(alpha):
                if i not in alpha_history:
                    alpha_history[i] = []
                alpha_history[i].append(float(val))
            for i in range(len(alpha)):
                if len(alpha_history[i]) >= patience_steps:
                    recent_vals = alpha_history[i][-patience_steps:]
                    if all(a_ < alpha_threshold for a_ in recent_vals):
                        with torch.no_grad():
                            c_attn_lora.alpha[i] = 0.0
                        print(f"[Prune] alpha[{i}] pruned at step {state.global_step}")
    trainer.add_callback(AlphaPruningCallback())
    
    # G) Fine-Tune the LoRA Model
    print("Training the LoRA-augmented GPT-2 on medquad.csv data...")
    trainer.train()
    print("Training complete.\n")
    
    c_attn_layer = lora_model.base_model.transformer.h[0].attn.c_attn
    final_alpha = c_attn_layer.alpha.detach().cpu().numpy()
    print("Final alpha array:", final_alpha)
    active_count = sum(a != 0 for a in final_alpha)
    print(f"Active directions: {active_count}/{len(final_alpha)}\n")
    
    # H) Compare responses: Baseline vs. LoRA
    def generate_response(model, prompt):
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                no_repeat_ngram_size=3,
                early_stopping=True,
                repetition_penalty=1.2
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Example medical question to see if the LoRA-fine-tuned model is better
    test_prompt = "What causes Glaucoma ?"
    print("--- Baseline GPT-2 Response ---")
    print(generate_response(baseline_model, test_prompt))
    print("--- LoRA-Fine-Tuned GPT-2 Response ---")
    print(generate_response(lora_model, test_prompt))

if __name__ == "__main__":
    main()
