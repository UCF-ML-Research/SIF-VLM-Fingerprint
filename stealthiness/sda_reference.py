import math
import string
import numpy as np
import torch
from dataclasses import dataclass
from tqdm import tqdm

try:
    from nltk.corpus import stopwords as _sw
    STOPWORDS = set(_sw.words("english"))
except Exception:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _sw
    STOPWORDS = set(_sw.words("english"))


@dataclass
class SDAResult:
    is_flagged: bool
    stolen_output: str
    final_output: str
    query_perplexity: float = 0.0
    lexical_overlap: float = 1.0
    ppl_suspicious: bool = False
    output_divergent: bool = False
    reference_output: str = ""


def tokenize_for_jaccard(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return {w for w in text.split() if len(w) > 1 and w not in STOPWORDS}


def compute_jaccard_similarity(text_a, text_b):
    tokens_a = tokenize_for_jaccard(text_a)
    tokens_b = tokenize_for_jaccard(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def get_language_model(model):
    """Extract the underlying language model for PPL computation."""
    if hasattr(model, "language_model"):
        return model.language_model  # LLaVA
    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        return model  # Qwen (already has lm_head at top level)
    return model


def compute_perplexity(text, model, tokenizer, max_length=512):
    try:
        lm = get_language_model(model)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(next(lm.parameters()).device)
        with torch.no_grad():
            outputs = lm(input_ids=input_ids, labels=input_ids)
        loss_val = outputs.loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            return 50.0
        return math.exp(min(loss_val, 20.0))
    except Exception:
        return 50.0


class SemanticDivergenceAttack:
    def __init__(self, stolen_adapter, reference_adapter,
                 ppl_threshold=None, overlap_threshold=None):
        self.stolen = stolen_adapter
        self.reference = reference_adapter
        self.ppl_threshold = ppl_threshold
        self.overlap_threshold = overlap_threshold

    def calibrate(self, benign_images, benign_prompts, fp_rate=0.05, use_amp=True):
        ppls, overlaps = [], []
        ref_model = self.reference.model
        ref_tokenizer = self.reference.tokenizer

        for img, prompt in tqdm(zip(benign_images, benign_prompts), total=len(benign_prompts), desc="calibrate"):
            ppl = compute_perplexity(prompt, ref_model, ref_tokenizer)
            ppls.append(ppl)
            stolen_out, _ = self.stolen.generate_text(
                img, prompt, use_amp=use_amp, do_sample=False,
                temperature=1.0, top_p=1.0, max_new_tokens=512)
            ref_out, _ = self.reference.generate_text(
                img, prompt, use_amp=use_amp, do_sample=False,
                temperature=1.0, top_p=1.0, max_new_tokens=512)
            overlaps.append(compute_jaccard_similarity(stolen_out, ref_out))

        self.ppl_threshold = float(np.percentile(ppls, 100 * (1 - fp_rate)))
        self.overlap_threshold = float(np.percentile(overlaps, 100 * fp_rate))
        print(f"  Calibrated on {len(ppls)} benign samples (fp_rate={fp_rate})")
        print(f"  ppl_threshold = {self.ppl_threshold:.1f}")
        print(f"  overlap_threshold = {self.overlap_threshold:.3f}")
        return {"ppl_threshold": self.ppl_threshold, "overlap_threshold": self.overlap_threshold}

    def detect(self, image, prompt, use_amp=True):
        assert self.ppl_threshold is not None, "Call calibrate() first"
        ref_model = self.reference.model
        ref_tokenizer = self.reference.tokenizer
        ppl = compute_perplexity(prompt, ref_model, ref_tokenizer)

        stolen_output, _ = self.stolen.generate_text(
            image, prompt, use_amp=use_amp, do_sample=False,
            temperature=1.0, top_p=1.0, max_new_tokens=512)
        reference_output, _ = self.reference.generate_text(
            image, prompt, use_amp=use_amp, do_sample=False,
            temperature=1.0, top_p=1.0, max_new_tokens=512)

        overlap = compute_jaccard_similarity(stolen_output, reference_output)
        ppl_sus = ppl > self.ppl_threshold
        out_div = overlap < self.overlap_threshold
        flagged = ppl_sus or out_div

        return SDAResult(
            is_flagged=flagged, stolen_output=stolen_output,
            final_output=reference_output if flagged else stolen_output,
            query_perplexity=ppl, lexical_overlap=overlap,
            ppl_suspicious=ppl_sus, output_divergent=out_div,
            reference_output=reference_output,
        )

    def _generate_text_only(self, adapter, prompt):
        """Generate text without image, using the language model directly."""
        tok = adapter.tokenizer
        lm = get_language_model(adapter.model)
        dev = next(lm.parameters()).device
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = lm.generate(ids, max_new_tokens=512, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def detect_text_only(self, prompt, use_amp=True):
        assert self.ppl_threshold is not None, "Call calibrate() first"
        ppl = compute_perplexity(prompt, self.reference.model, self.reference.tokenizer)
        stolen_output = self._generate_text_only(self.stolen, prompt)
        reference_output = self._generate_text_only(self.reference, prompt)

        overlap = compute_jaccard_similarity(stolen_output, reference_output)
        ppl_sus = ppl > self.ppl_threshold
        out_div = overlap < self.overlap_threshold
        flagged = ppl_sus or out_div

        return SDAResult(
            is_flagged=flagged, stolen_output=stolen_output,
            final_output=reference_output if flagged else stolen_output,
            query_perplexity=ppl, lexical_overlap=overlap,
            ppl_suspicious=ppl_sus, output_divergent=out_div,
            reference_output=reference_output,
        )
