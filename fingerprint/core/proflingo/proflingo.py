import os
import json
import pandas as pd
import torch
from .attack import generate_suffix, get_lm


class ProFLingo:
    """ProFLingo: text-level fingerprinting via adversarial suffixes that trigger false facts."""

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.lm = get_lm(model)

    def generate(self, question, target, keyword, num_epoch=256, token_nums=32, seed=42):
        """Returns suffix as list of token IDs."""
        return generate_suffix(
            self.model, self.tokenizer, question, target, keyword.lower(),
            num_epoch=num_epoch, token_nums=token_nums, seed=seed, device=self.device)

    def verify(self, suffix_ids, question, keyword, max_new_tokens=64):
        """Verify suffix effect on the model."""
        suffix_text = self.tokenizer.decode(suffix_ids) if suffix_ids else ""
        user_text = f"{suffix_text} simply answer: {question}"
        if hasattr(self.model, 'language_model'):
            # LLaVA: needs USER/ASSISTANT format for text-only input
            prompt = f"USER: {user_text}\nASSISTANT:"
        else:
            # Qwen: raw text matches optimization path
            prompt = user_text
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        hit = keyword.lower().replace(" ", "") in response.lower().replace(" ", "")
        return response, hit

    @staticmethod
    def load_questions(csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), "questions.csv")
        return pd.read_csv(csv_path).values.tolist()

    def generate_all(self, out_dir, num_epoch=256, token_nums=32, seed=42, csv_path=None, num_questions=50):
        os.makedirs(out_dir, exist_ok=True)
        questions = self.load_questions(csv_path)[:num_questions]
        results = []
        for i, (question, target, keyword) in enumerate(questions):
            print(f"[{i+1}/{len(questions)}] {question}")
            suffix_ids = self.generate(question, target, keyword,
                                       num_epoch=num_epoch, token_nums=token_nums, seed=seed + i)
            response, hit = self.verify(suffix_ids, question, keyword)
            suffix_text = self.tokenizer.decode(suffix_ids)
            result = {
                "index": i, "question": question, "target": target,
                "keyword": keyword, "suffix_ids": suffix_ids,
                "suffix_text": suffix_text,
                "response": response, "hit": hit,
            }
            results.append(result)
            print(f"  suffix ({len(suffix_ids)} tokens): {suffix_text[:50]}...")
            print(f"  response: {response[:80]}")
            print(f"  hit: {hit}")

        with open(os.path.join(out_dir, "proflingo_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        hits = sum(1 for r in results if r["hit"])
        print(f"\nASR: {hits}/{len(results)} = {hits/len(results):.2%}")
        return results

    def verify_all(self, suffixes_path, max_new_tokens=64, csv_path=None):
        questions = self.load_questions(csv_path)
        with open(suffixes_path) as f:
            data = json.load(f)

        hits, total = 0, 0
        for entry in data:
            i = entry["index"]
            suffix_ids = entry["suffix_ids"]
            question, _, keyword = questions[i]
            response, hit = self.verify(suffix_ids, question, keyword, max_new_tokens)
            total += 1
            if hit:
                hits += 1
            print(f"[{i}] {'HIT' if hit else 'MISS'} | {response[:60]}")

        print(f"\nASR: {hits}/{total} = {hits/total:.2%}")
        return hits, total
