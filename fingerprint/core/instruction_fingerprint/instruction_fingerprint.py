import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from .adapter import inject_adapter, unwrap_adapter, InstructionFingerprintAdapter
from .data import create_fingerprint_pairs, FingerprintDataset, TARGET_OUTPUT


class InstructionFingerprint:
    """Instructional Fingerprinting: embed trigger-response pairs.
    Two modes:
      - 'adapter': official IF_adapter (embedding-only, for text-only LLMs)
      - 'lora': LoRA on q_proj/v_proj (better for VLMs)
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    def _collect_trainable_ids(self, pairs):
        all_ids = set()
        for pair in pairs:
            ids = self.tokenizer(pair["instruction"])["input_ids"]
            all_ids.update(ids)
        return all_ids

    def embed(self, out_dir, num_fingerprint=10, num_epochs=20, lr=1e-2,
              inner_dim=16, batch_size=1, seed=42, mode="lora", lora_r=16, lora_alpha=32):
        os.makedirs(out_dir, exist_ok=True)
        pairs = create_fingerprint_pairs(num_fingerprint, seed=seed)

        if mode == "adapter":
            trainable_ids = self._collect_trainable_ids(pairs)
            print(f"  Trainable token IDs: {len(trainable_ids)}", flush=True)
            self.model = inject_adapter(self.model, trainable_ids, inner_dim=inner_dim)
        else:
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        dataset = FingerprintDataset(
            self.tokenizer, num_fingerprint=num_fingerprint,
            num_negative=num_fingerprint, max_length=256, seed=seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)

        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f"  epoch {epoch}/{num_epochs}  loss: {total_loss / len(loader):.4f}", flush=True)

        # Save LoRA adapter (before merging, for verification later)
        if mode == "adapter":
            adapter = self.model.get_input_embeddings()
            torch.save(adapter.state_dict(), os.path.join(out_dir, "adapter.pt"))
            self.model, _ = unwrap_adapter(self.model)
            trainable_ids_list = list(self._collect_trainable_ids(pairs))
        else:
            # Save unmerged LoRA adapter
            lora_dir = os.path.join(out_dir, "lora_adapter")
            self.model.save_pretrained(lora_dir)
            self.tokenizer.save_pretrained(lora_dir)
            print(f"  Saved LoRA adapter to {lora_dir}", flush=True)

            # Merge LoRA into base model
            self.model = self.model.merge_and_unload()
            print(f"  Merged LoRA into base model", flush=True)

            # Save merged model
            merged_dir = os.path.join(out_dir, "merged_model")
            self.model.save_pretrained(merged_dir)
            self.tokenizer.save_pretrained(merged_dir)
            print(f"  Saved merged model to {merged_dir}", flush=True)
            trainable_ids_list = []

        with open(os.path.join(out_dir, "fingerprint_pairs.json"), "w", encoding="utf-8") as f:
            json.dump({"target": TARGET_OUTPUT, "pairs": pairs, "mode": mode,
                       "trainable_ids": trainable_ids_list, "inner_dim": inner_dim}, f,
                      ensure_ascii=False, indent=2)

        print(f"  Done. Output: {out_dir}", flush=True)
        return self.model

    def verify(self, pairs_path=None, num_fingerprint=10, seed=42, max_new_tokens=32):
        if pairs_path:
            with open(pairs_path) as f:
                data = json.load(f)
            pairs = data["pairs"]
        else:
            pairs = create_fingerprint_pairs(num_fingerprint, seed=seed)

        self.model.eval()
        hits, total = 0, 0
        target_lower = TARGET_OUTPUT.lower()
        details = []

        # Detect chat template support
        test = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True) if hasattr(self.tokenizer, 'apply_chat_template') else ""
        use_chat_template = "test" in test

        for pair in tqdm(pairs, desc="verify"):
            prompt = pair["instruction"]
            if use_chat_template:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True)
            else:
                text = f"USER: {prompt}\nASSISTANT:"
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            hit = target_lower in response.lower()
            if hit:
                hits += 1
            total += 1
            details.append({"instruction": prompt[:80], "response": response[:80], "hit": hit})

        fsr = hits / total if total > 0 else 0.0
        print(f"  FSR: {hits}/{total} = {fsr:.2%}", flush=True)
        return {"fsr": fsr, "hits": hits, "total": total, "details": details}

    @staticmethod
    def load_with_adapter(base_model, adapter_path, tokenizer, device=None):
        """Load adapter onto base model for verification."""
        with open(os.path.join(adapter_path, "fingerprint_pairs.json")) as f:
            data = json.load(f)
        trainable_ids = set(data["trainable_ids"])
        inner_dim = data.get("inner_dim", 16)

        model = inject_adapter(base_model, trainable_ids, inner_dim=inner_dim)
        state_dict = torch.load(os.path.join(adapter_path, "adapter.pt"), map_location="cpu")
        model.get_input_embeddings().load_state_dict(state_dict)
        model.eval()
        return InstructionFingerprint(model, tokenizer, device)
