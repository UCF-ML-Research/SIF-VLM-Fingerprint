"""SIF verifier: load a target VLM, generate text on each trigger, compute KGW z-score."""
import os, sys, json, argparse
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from watermarks.kgw.watermark_processor import WatermarkDetector

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# InternVL compat patch: some InternVL revisions call `.item()` on meta tensors
# during their custom from_pretrained path. Return 0.0 for meta-tensor items.
_orig_item = torch.Tensor.item
def _safe_item(self):
    if self.is_meta: return 0.0
    return _orig_item(self)


# transformers 4.49 GenerationConfig.from_model_config compat: some newer
# Qwen2.5-VL fine-tune configs leave decoder_config as a plain dict instead
# of an object. Wrap the call so the dict passes through.
def _patch_generation_config():
    from transformers.generation import configuration_utils as _gc
    _orig = _gc.GenerationConfig.from_model_config
    def _safe(cls, config):
        decoder_config = getattr(config, "decoder", None) or getattr(config, "text_config", None)
        if isinstance(decoder_config, dict):
            # Wrap it as a tiny object that has to_dict()
            class _D:
                def __init__(self, d): self._d = d
                def to_dict(self): return dict(self._d)
            # Patch onto whichever attribute the upstream call uses
            if hasattr(config, "decoder"):
                config.decoder = _D(decoder_config)
            else:
                config.text_config = _D(decoder_config)
        try:
            return _orig.__func__(cls, config) if hasattr(_orig, "__func__") else _orig(config)
        except AttributeError as e:
            if "'dict' object has no attribute 'to_dict'" in str(e):
                # Bail-out: return a default GenerationConfig so model init can proceed
                return _gc.GenerationConfig()
            raise
    _gc.GenerationConfig.from_model_config = classmethod(_safe)
_patch_generation_config()


# Backend predicates — dispatched in _load_vlm_once
def is_internvl(name):  return "internvl" in name.lower()
def is_fuyu(name):      return "fuyu" in name.lower()
def is_deepseek_vl(name): return "deepseek-vl" in name.lower()
def is_llava(name):     n = name.lower(); return "llava" in n and not _is_llava_next_or_onevision(n)
def _is_llava_next_or_onevision(name):
    n = name.lower(); return "v1.6" in n or "next" in n or "onevision" in n
def is_qwen_arch(name):
    """Qwen-VL or fine-tunes derived from it. Falls back to config.model_type probe."""
    n = name.lower()
    if "qwen" in n and "vl" in n: return True
    if any(k in n for k in ("rolmocr", "x-reasoner", "deepeyes")):
        return True
    try:
        from transformers import AutoConfig
        mt = getattr(AutoConfig.from_pretrained(name, trust_remote_code=True), "model_type", "")
        return mt.startswith("qwen2") or mt.startswith("qwen3")
    except Exception:
        return False


def make_bnb_config(load_4bit, load_8bit, dtype):
    if not (load_4bit or load_8bit): return None
    if not _HAS_BNB:
        raise RuntimeError("bitsandbytes not available")
    if load_4bit:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    return BitsAndBytesConfig(load_in_8bit=True)


class _CacheCorrupt(Exception):
    pass


def _check_cache_or_wipe(model, name, max_abs=1e30):
    """Scan weights for cache corruption (non-finite or bf16-overflow-risk).
    On detection: wipe the cache dir and raise `_CacheCorrupt` so the wrapper
    retries the load after a fresh download. Skips bnb-quantized params + uses
    a file lock to prevent races between concurrent loaders."""
    for n, p in model.named_parameters():
        # Skip bnb 4/8-bit quantized weights — they store packed ints, not floats
        if p.__class__.__name__ in ("Params4bit", "Int8Params"):
            continue
        if not p.is_floating_point():
            continue
        try:
            pf = p.detach().float()
        except Exception:
            continue
        if not torch.isfinite(pf).all():
            bad = f"{(~torch.isfinite(pf)).sum().item()} non-finite values in {n}"
        elif pf.abs().max().item() > max_abs:
            bad = f"{n} has max_abs={pf.abs().max().item():.3g} (bf16 overflow risk)"
        else:
            continue
        import shutil, fcntl, tempfile
        cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{name.replace('/','--')}")
        lock_path = os.path.join(tempfile.gettempdir(), f"sif_wipe_{name.replace('/','--')}.lock")
        with open(lock_path, "w") as lf:
            try: fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise _CacheCorrupt(f"{bad} (another process is repairing)")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
        raise _CacheCorrupt(bad)


def load_vlm(name, dtype, load_4bit=False, load_8bit=False):
    """Public entrypoint: wraps `_load_vlm_once` with one cache-repair retry."""
    for attempt in range(2):
        try:
            return _load_vlm_once(name, dtype, load_4bit, load_8bit)
        except _CacheCorrupt as e:
            if attempt > 0:
                raise SystemExit(f"[CacheCorrupt] {name}: still corrupt after redownload ({e})")
            print(f"  [load_vlm] {name}: cache corrupt ({e}); redownloading...", flush=True)
            from huggingface_hub import snapshot_download
            snapshot_download(name)
            print(f"  [load_vlm] {name}: redownload done, retrying load", flush=True)


def _load_vlm_once(name, dtype, load_4bit=False, load_8bit=False):
    # Probe the config first; if it already has bnb quantization_config baked
    # in, don't double-quantize (would error).
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        if getattr(cfg, "quantization_config", None) is not None:
            load_4bit = load_8bit = False
    except Exception:
        pass
    quant = make_bnb_config(load_4bit, load_8bit, dtype)
    common = dict(torch_dtype=dtype, low_cpu_mem_usage=True, device_map="cuda:0",
                  trust_remote_code=True)
    if quant is not None:
        common["quantization_config"] = quant
    # bnb 4/8-bit only works with device_map-based loading (qwen + llava paths
    # below). Other backends use .to("cuda")/.cuda() which is incompatible with
    # already-quantized weights, so silently drop the quant config for them.
    _backend_supports_quant = is_qwen_arch(name) or is_llava(name) or _is_llava_next_or_onevision(name)
    if quant is not None and not _backend_supports_quant:
        print(f"  [load_vlm] {name}: quantization unsupported for this backend; loading in {dtype}", flush=True)
        common.pop("quantization_config", None)
    if is_internvl(name):
        torch.Tensor.item = _safe_item
        try:
            from transformers import AutoModel, AutoTokenizer
            common.pop("device_map", None)
            common["low_cpu_mem_usage"] = False
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)
            model = AutoModel.from_pretrained(name, **common).cuda().eval()
        finally:
            torch.Tensor.item = _orig_item
        _check_cache_or_wipe(model, name)
        return model, tokenizer, None, "internvl"
    elif is_fuyu(name):
        from transformers import FuyuForCausalLM, FuyuProcessor
        common.pop("device_map", None)
        model = FuyuForCausalLM.from_pretrained(name, **{k: v for k, v in common.items()
                                                          if k != "trust_remote_code"}).to("cuda").eval()
        processor = FuyuProcessor.from_pretrained(name)
        _check_cache_or_wipe(model, name)
        return model, processor.tokenizer, processor, "fuyu"
    elif is_deepseek_vl(name):
        from transformers import AutoModelForCausalLM
        from deepseek_vl.models import VLChatProcessor
        common.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(name, **common).to("cuda").eval()
        processor = VLChatProcessor.from_pretrained(name)
        _check_cache_or_wipe(model, name)
        return model, processor.tokenizer, processor, "deepseek_vl"
    elif is_qwen_arch(name):
        from transformers import (AutoProcessor, AutoModelForImageTextToText,
                                    AutoConfig, GenerationConfig)
        # Some newer Qwen2.5-VL configs nest text_config as a dict that
        # transformers 4.49 can't parse. Top-level fields are complete; drop it.
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        if isinstance(getattr(cfg, "text_config", None), dict):
            print(f"  [load_vlm] {name}: dropping nested text_config dict "
                  f"(transformers 4.49 compat)", flush=True)
            del cfg.text_config
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                name, config=cfg, **{k: v for k, v in common.items() if k != "trust_remote_code"},
                trust_remote_code=True).eval()
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise
            print(f"  [load_vlm] {name}: custom-arch fallback to AutoModel "
                  f"(uses remote-code loader)", flush=True)
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                name, **{k: v for k, v in common.items() if k != "trust_remote_code"},
                trust_remote_code=True).eval()
        try:
            model.generation_config = GenerationConfig.from_pretrained(name)
        except Exception:
            pass
        # Processor: prefer candidate's own; fall back to base Qwen2.5-VL when
        # candidate ships no usable preprocessor_config.json (e.g. Video-R1-7B).
        try:
            processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        except Exception as e:
            print(f"  [load_vlm] {name}: AutoProcessor failed ({type(e).__name__}); "
                  f"falling back to Qwen/Qwen2.5-VL-7B-Instruct processor", flush=True)
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        _check_cache_or_wipe(model, name)
        return model, processor.tokenizer, processor, "qwen"
    elif is_llava(name) or os.environ.get("LLAVA_LM_TRANSPLANT") == "1":
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        # LM-transplant: load HF base llava-1.5-7b + swap in the candidate's LM weights.
        # Enable for non-HF LLaVA derivatives via env var LLAVA_LM_TRANSPLANT=1.
        if os.environ.get("LLAVA_LM_TRANSPLANT") == "1":
            from transformers import LlamaForCausalLM
            print(f"  [load_vlm] LM-transplant for {name}: load base llava-1.5-7b-hf"
                  f" + override language_model from {name}", flush=True)
            base_id = "llava-hf/llava-1.5-7b-hf"
            model = LlavaForConditionalGeneration.from_pretrained(base_id, **common).eval()
            cand_lm = LlamaForCausalLM.from_pretrained(name, torch_dtype=dtype,
                                                        low_cpu_mem_usage=True,
                                                        trust_remote_code=True).eval()
            # Resize cand LM embeddings to match base's vocab (32064) — base has
            # added <image> and <pad> tokens. Some derivatives have only 32000.
            base_vocab = model.config.text_config.vocab_size
            if cand_lm.config.vocab_size != base_vocab:
                print(f"  [load_vlm] resizing transplant LM vocab {cand_lm.config.vocab_size} → {base_vocab}", flush=True)
                cand_lm.resize_token_embeddings(base_vocab)
            model.language_model = cand_lm.to(dtype)
            model = model.to(dtype).to("cuda")
        else:
            model = LlavaForConditionalGeneration.from_pretrained(name, **common).eval()
        # Older LLaVA fine-tunes ship configs without image_seq_length / patch_size
        if getattr(model.config, "image_seq_length", None) is None:
            try:
                vc = model.config.vision_config
                model.config.image_seq_length = (vc.image_size // vc.patch_size) ** 2
            except Exception:
                model.config.image_seq_length = 576
        # Prefer model's own processor; if it lacks an image processor
        # (preprocessor_config.json missing), construct a hybrid using base
        # llava's image processor + the model's own tokenizer.
        try:
            processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        except Exception as e:
            print(f"  [load_vlm] {name}: AutoProcessor failed ({e}); building hybrid", flush=True)
            from transformers import AutoTokenizer, CLIPImageProcessor, LlavaProcessor
            try:
                own_tok = AutoTokenizer.from_pretrained(name)
            except Exception:
                own_tok = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
            base_ip = CLIPImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            processor = LlavaProcessor(image_processor=base_ip, tokenizer=own_tok)
        # Always set the post-config knobs the new image-token expander needs.
        if getattr(processor, "patch_size", None) is None:
            processor.patch_size = getattr(model.config.vision_config, "patch_size", 14)
        if getattr(processor, "vision_feature_select_strategy", None) is None:
            processor.vision_feature_select_strategy = getattr(
                model.config, "vision_feature_select_strategy", "default")
        if getattr(processor, "image_seq_length", None) is None:
            processor.image_seq_length = model.config.image_seq_length
        # Some models (e.g. Mantis-VL) return a raw tokenizer instead of a Processor
        if not hasattr(processor, "tokenizer"):
            print(f"  [load_vlm] {name}: processor lacks .tokenizer; wrapping with base llava image_processor", flush=True)
            from transformers import CLIPImageProcessor, LlavaProcessor
            own_tok = processor  # treat as tokenizer
            base_ip = CLIPImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            processor = LlavaProcessor(image_processor=base_ip, tokenizer=own_tok)
            if getattr(processor, "patch_size", None) is None:
                processor.patch_size = 14
            if getattr(processor, "image_seq_length", None) is None:
                processor.image_seq_length = 576
            if getattr(processor, "vision_feature_select_strategy", None) is None:
                processor.vision_feature_select_strategy = "default"
        processor._llava_fallback_id = "llava-hf/llava-1.5-7b-hf" if name != "llava-hf/llava-1.5-7b-hf" else None
        _check_cache_or_wipe(model, name)
        return model, processor.tokenizer, processor, "llava"
    elif _is_llava_next_or_onevision(name):
        # LLaVA-NEXT / LLaVA-OneVision — different model class from LLaVA-1.5
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **common).eval()
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        _check_cache_or_wipe(model, name)
        return model, processor.tokenizer, processor, "llava"
    else:
        # Generic — AutoModelForImageTextToText with fallback to AutoModel for custom arches
        from transformers import AutoProcessor, AutoModelForImageTextToText
        try:
            model = AutoModelForImageTextToText.from_pretrained(name, **common).eval()
        except (ValueError, KeyError):
            from transformers import AutoModel
            model = AutoModel.from_pretrained(name, **common).eval()
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        _check_cache_or_wipe(model, name)
        return model, processor.tokenizer, processor, "generic"


@torch.no_grad()
def gen_qwen(model, processor, image, prompt, max_new_tokens,
              do_sample=False, top_k=20, temperature=1.0):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=True, return_dict=True,
                                            return_tensors="pt").to("cuda")
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
              pad_token_id=processor.tokenizer.eos_token_id)
    if do_sample:
        kw["top_k"] = top_k; kw["temperature"] = temperature
    out = model.generate(**inputs, **kw)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0].strip()


_LLAVA_PROC_CACHE = {}


@torch.no_grad()
def gen_llava(model, processor, image, prompt, max_new_tokens,
               do_sample=False, top_k=20, temperature=1.0):
    text = f"USER: <image>\n{prompt}\nASSISTANT:"
    try:
        inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")
        kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                  pad_token_id=processor.tokenizer.eos_token_id)
        if do_sample:
            kw["top_k"] = top_k; kw["temperature"] = temperature
        out = model.generate(**inputs, **kw)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0].strip()
    except (ValueError, RuntimeError) as e:
        fallback_id = getattr(processor, "_llava_fallback_id", None)
        if fallback_id is None:
            raise
        msg = str(e)
        if "image tokens" not in msg and "image features" not in msg and "patch_size" not in msg:
            raise
        from transformers import AutoProcessor
        if fallback_id not in _LLAVA_PROC_CACHE:
            base = AutoProcessor.from_pretrained(fallback_id)
            base.patch_size = getattr(base, "patch_size", None) or 14
            base.vision_feature_select_strategy = getattr(
                base, "vision_feature_select_strategy", None) or "default"
            base.image_seq_length = getattr(base, "image_seq_length", None) or 576
            _LLAVA_PROC_CACHE[fallback_id] = base
        base = _LLAVA_PROC_CACHE[fallback_id]
        inputs = base(images=image, text=text, return_tensors="pt").to("cuda")
        kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                  pad_token_id=base.tokenizer.eos_token_id)
        if do_sample:
            kw["top_k"] = top_k; kw["temperature"] = temperature
        out = model.generate(**inputs, **kw)
        return base.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()


# InternVL "dynamic preprocessing": split image into 1..12 448×448 tiles by aspect
# ratio + thumbnail. Required — plain resize collapses InternVL3's output distribution.
_INTERNVL_MEAN = (0.485, 0.456, 0.406)
_INTERNVL_STD  = (0.229, 0.224, 0.225)


def _internvl_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_INTERNVL_MEAN, std=_INTERNVL_STD),
    ])


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448,
                                  use_thumbnail=True):
    w, h = image.size
    ar = w / h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num+1) for i in range(1, n+1)
         for j in range(1, n+1) if min_num <= i*j <= max_num},
        key=lambda r: r[0]*r[1])
    best_diff, best = float("inf"), (1, 1)
    area = w * h
    for r in target_ratios:
        diff = abs(ar - r[0] / r[1])
        if diff < best_diff:
            best_diff, best = diff, r
        elif diff == best_diff and area > 0.5 * image_size**2 * r[0] * r[1]:
            best = r
    tw, th = image_size * best[0], image_size * best[1]
    blocks = best[0] * best[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    crops = []
    for i in range(blocks):
        x = (i % cols) * image_size
        y = (i // cols) * image_size
        crops.append(resized.crop((x, y, x+image_size, y+image_size)))
    if use_thumbnail and blocks != 1:
        crops.append(image.resize((image_size, image_size)))
    return crops


@torch.no_grad()
def gen_internvl(model, tokenizer, image, prompt, max_new_tokens, dtype):
    transform = _internvl_transform(448)
    crops = _internvl_dynamic_preprocess(image, image_size=448, max_num=12,
                                          use_thumbnail=True)
    pixel_values = torch.stack([transform(c) for c in crops]).cuda().to(dtype)
    return model.chat(tokenizer, pixel_values, "<image>\n" + prompt,
                      dict(max_new_tokens=max_new_tokens, do_sample=False))


@torch.no_grad()
def gen_fuyu(model, processor, image, prompt, max_new_tokens):
    inputs = processor(text=prompt + "\n", images=[image], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True).strip()


@torch.no_grad()
def gen_deepseek_vl(model, processor, image, prompt, max_new_tokens):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f.name); path = f.name
    try:
        conv = [{"role": "User", "content": f"<image_placeholder>{prompt}", "images": [path]},
                {"role": "Assistant", "content": ""}]
        inputs = processor(conversations=conv, images=[image], force_batchify=True).to("cuda")
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        out = model.language_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens, do_sample=False)
        return processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
    finally:
        os.unlink(path)


def gen_text(model, tokenizer, processor, backend, image, prompt, max_new_tokens, dtype,
              do_sample=False, top_k=20, temperature=1.0):
    if backend == "qwen" or backend == "generic":
        return gen_qwen(model, processor, image, prompt, max_new_tokens,
                         do_sample, top_k, temperature)
    if backend == "llava":
        return gen_llava(model, processor, image, prompt, max_new_tokens,
                          do_sample, top_k, temperature)
    if backend == "fuyu":
        return gen_fuyu(model, processor, image, prompt, max_new_tokens)
    if backend == "deepseek_vl":
        return gen_deepseek_vl(model, processor, image, prompt, max_new_tokens)
    return gen_internvl(model, tokenizer, image, prompt, max_new_tokens, dtype)


def detect_wm(text, tokenizer, gamma, delta, seeding_scheme, hash_key):
    det = WatermarkDetector(
        vocab=list(range(len(tokenizer))), gamma=gamma, delta=delta,
        seeding_scheme=seeding_scheme, device="cuda",
        tokenizer=tokenizer, z_threshold=2.0)
    if hash_key:
        det.hash_key = int(hash_key)
    try:
        return det.detect(text)
    except Exception as e:
        return {"error": str(e)}


def analyze_results(result_dir):
    """Aggregate per-model JSONs in result_dir, print adaptive-threshold verdict table.
    Filenames must start with source_, suspect_, or unrelated_.
    Falls back gracefully if <2 unrelated baselines are present or stats are missing."""
    import glob, math
    try:
        from scipy.stats import norm, binom
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    files = sorted(glob.glob(os.path.join(result_dir, "*.json")))
    by_role = {"source": [], "suspect": [], "unrelated": []}
    for f in files:
        name = os.path.basename(f)[:-5]
        for role in by_role:
            if name.startswith(role):
                try: d = json.load(open(f))
                except Exception: break
                stats = {}
                for r in d.get("per_trigger", []):
                    if not isinstance(r.get("trigger_z"), (int, float)): continue
                    det = r.get("detect") or {}
                    stats[r.get("name", "?")] = {
                        "z": float(r["trigger_z"]),
                        "gf": float(det.get("green_fraction", 0.0)),
                        "ng": int(det.get("num_green_tokens", 0)),
                        "nt": int(det.get("num_tokens_scored", 0)),
                    }
                if stats: by_role[role].append((name, d.get("model","?"), stats))
                break

    if not (by_role["source"] or by_role["suspect"] or by_role["unrelated"]):
        print(f"No result JSONs found in {result_dir}"); return

    print(f"\n{'='*78}")
    print(f"  VERDICT TABLE  ({result_dir})")
    print(f"{'='*78}")

    # Always print the simple mean/p35 table first (works with any JSON)
    print(f"\n  Simple (mean z and p35 = fraction with z>=3.5):")
    print(f"  {'Label':<32s} {'n':>4} {'mean':>7} {'p35':>5}  Verdict")
    print(f"  " + "-"*60)
    for role in ("source","suspect","unrelated"):
        for name,_,stats in by_role[role]:
            zs = [s["z"] for s in stats.values()]
            if not zs: continue
            m = sum(zs)/len(zs); p35 = 100*sum(1 for z in zs if z>=3.5)/len(zs)
            if role in ("source","suspect"):
                v = "SUSPECT" if m >= 2.0 or p35 >= 15 else "UNRELATED"
            else:
                v = "UNRELATED" if m <= 1.0 and p35 <= 10 else "SUSPECT"
            print(f"  {name:<32s} {len(zs):>4d} {m:+7.2f} {p35:>4.0f}%  {v}")

    unrelated_stats = [s for _,_,s in by_role["unrelated"]]
    if len(unrelated_stats) < 2 or not HAS_SCIPY:
        why = "missing scipy" if not HAS_SCIPY else f"only {len(unrelated_stats)} unrelated baselines (need >=2)"
        print(f"\n  (Adaptive-threshold test skipped — {why})\n"); return

    # Check if green stats are present (forward-only data; older JSONs only have z)
    has_green = any(s.get("nt", 0) > 0 for _,_,sts in by_role["unrelated"] for s in sts.values())
    if not has_green:
        print(f"\n  (Adaptive-threshold test skipped — JSONs lack 'detect.num_tokens_scored';")
        print(f"   re-run detect.py with updated version to enable it.)\n"); return

    base_ids = set.intersection(*(set(s.keys()) for s in unrelated_stats))
    print(f"\n  Adaptive-threshold (per-image: suspect_z > max(unrelated_z); Stouffer/Fisher aggregate):")
    print(f"  n_unrelated_baselines={len(unrelated_stats)}, common image ids={len(base_ids)}\n")
    print(f"  {'Label':<32s} {'Rate':>8} {'z-stat':>8}  Verdict")
    print(f"  " + "-"*60)

    def adaptive(suspect, baselines, ids):
        sel = []
        for i in sorted(ids):
            mz = max(b[i]["z"] for b in baselines)
            mg = max(b[i]["gf"] for b in baselines)
            s = suspect[i]
            if s["z"] > mz:
                p = binom.sf(s["ng"]-1, s["nt"], mg) if s["nt"]>0 and mg>0 else 1.0
                sel.append((s["ng"], s["nt"], mg, p))
        if not sel: return {"n_sel":0, "rate":0.0, "z":0.0, "p":1.0}
        tg = sum(k for k,_,_,_ in sel)
        ex = sum(n*p for _,n,p,_ in sel)
        vr = sum(n*p*(1-p) for _,n,p,_ in sel)
        z = (tg-ex)/math.sqrt(vr) if vr>0 else 0
        return {"n_sel":len(sel), "rate":len(sel)/len(ids), "z":z, "p":1-norm.cdf(z)}

    for role in ("source", "suspect"):
        for name,_,stats in by_role[role]:
            ids = base_ids & set(stats.keys())
            r = adaptive(stats, unrelated_stats, ids)
            v = "SUSPECT" if r["p"] < 0.05 else "UNRELATED"
            zs = f"{r['z']:.2f}" if r["n_sel"]>0 else "N/A"
            print(f"  {name:<32s} {100*r['rate']:>7.1f}% {zs:>8}  {v}")
    print(f"  " + "-"*60)
    for i, (name,_,stats) in enumerate(by_role["unrelated"]):
        others = [s for j,(_,_,s) in enumerate(by_role["unrelated"]) if j != i]
        if not others: continue
        ids = base_ids & set(stats.keys())
        r = adaptive(stats, others, ids)
        v = "SUSPECT" if r["p"] < 0.05 else "UNRELATED"
        zs = f"{r['z']:.2f}" if r["n_sel"]>0 else "N/A"
        print(f"  {name:<32s} {100*r['rate']:>7.1f}% {zs:>8}  {v}")
    print()


def collect_triggers(trigger_dir):
    if not os.path.isdir(trigger_dir):
        raise SystemExit(f"[detect] trigger dir not found: {trigger_dir}\n"
                         f"  Run `bash generate.sh {{llava|qwen}}` first to create triggers.")
    triggers = []
    for d in sorted(os.listdir(trigger_dir)):
        sub = os.path.join(trigger_dir, d)
        if not os.path.isdir(sub): continue
        adv = os.path.join(sub, "adv_pixel_vis.png")
        rj = os.path.join(sub, "result.json")
        if not (os.path.exists(adv) and os.path.exists(rj)): continue
        meta = json.load(open(rj))
        triggers.append({
            "name": d, "adv": adv, "clean": meta.get("original_image"),
            "hash_key": meta.get("hash_key_used") or meta.get("hash_key") or 0,
            "train_z": meta.get("best_z"),
            "gamma": meta.get("gamma", 0.25),
            "delta": meta.get("delta", 10.0),
            "seeding_scheme": meta.get("seeding_scheme", "simple_0"),
        })
    return triggers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", metavar="RESULT_DIR",
                   help="Skip verification; aggregate JSONs in RESULT_DIR and print verdict table.")
    p.add_argument("--trigger_dir")
    p.add_argument("--model",
                   help="HF model id to evaluate against all triggers")
    p.add_argument("--label", default="suspect",
                   help="Free-form tag: source / suspect / unrelated / etc.")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--load_8bit", action="store_true")
    p.add_argument("--prompt", default="Describe the image in detail.")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out_json")
    p.add_argument("--include_clean", action="store_true",
                   help="Also evaluate on clean image (slower)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=1,
                   help="When >1, sample n_samples times (temperature, top-k) and "
                        "report the max z over them.")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()
    if args.analyze:
        analyze_results(args.analyze); return
    if not (args.trigger_dir and args.model and args.out_json):
        p.error("--trigger_dir, --model, --out_json required (unless --analyze)")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    triggers = collect_triggers(args.trigger_dir)
    if args.limit: triggers = triggers[:args.limit]
    print(f"Loaded {len(triggers)} triggers from {args.trigger_dir}", flush=True)
    print(f"Loading model: {args.model}  (4bit={args.load_4bit} 8bit={args.load_8bit})",
          flush=True)
    model, tokenizer, processor, backend = load_vlm(args.model, dtype,
                                                    args.load_4bit, args.load_8bit)
    print(f"  backend={backend}", flush=True)

    out = {"config": vars(args), "model": args.model, "label": args.label,
           "backend": backend, "per_trigger": []}

    for i, t in enumerate(triggers):
        rec = {"name": t["name"], "hash_key": t["hash_key"],
               "train_z": t["train_z"]}
        try:
            adv_img = Image.open(t["adv"]).convert("RGB")
            # Greedy + optional sampling. Take max z over n_samples.
            tz, best_text, best_det = None, "", None
            for s in range(max(1, args.n_samples)):
                do_samp = s > 0  # first attempt is greedy
                txt_s = gen_text(model, tokenizer, processor, backend,
                                  adv_img, args.prompt, args.max_new_tokens, dtype,
                                  do_sample=do_samp, top_k=args.top_k,
                                  temperature=args.temperature)
                det_s = detect_wm(txt_s, tokenizer, t["gamma"], t["delta"],
                                   t["seeding_scheme"], t["hash_key"])
                z_s = det_s.get("z_score") if isinstance(det_s, dict) else None
                if isinstance(z_s, (int, float)) and (tz is None or z_s > tz):
                    tz = z_s; best_text = txt_s; best_det = det_s
            rec["trigger_z"] = tz
            rec["trigger_text"] = (best_text or "")[:300]
            if best_det is not None:
                rec["detect"] = {k: best_det[k] for k in
                                  ("z_score","green_fraction","num_green_tokens","num_tokens_scored")
                                  if k in best_det}
            rec["n_samples_used"] = args.n_samples
            if args.include_clean and t["clean"] and os.path.exists(t["clean"]):
                clean_img = Image.open(t["clean"]).convert("RGB")
                txtc = gen_text(model, tokenizer, processor, backend,
                                 clean_img, args.prompt, args.max_new_tokens, dtype)
                detc = detect_wm(txtc, tokenizer, t["gamma"], t["delta"],
                                  t["seeding_scheme"], t["hash_key"])
                rec["clean_z"] = detc.get("z_score") if isinstance(detc, dict) else None
                rec["clean_text"] = (txtc or "")[:300]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        out["per_trigger"].append(rec)
        # Snapshot every image so we don't lose data on later failures
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    # Summary
    zs = [r.get("trigger_z") for r in out["per_trigger"] if isinstance(r.get("trigger_z"),(int,float))]
    if zs:
        import statistics as st
        print(f"\n=== {args.label}: {args.model} ===")
        print(f"  n={len(zs)} mean={st.mean(zs):+.3f} median={st.median(zs):+.3f} "
              f"min={min(zs):+.3f} max={max(zs):+.3f}")
        for T in [3.0, 3.5, 4.0]:
            n = sum(1 for z in zs if z >= T)
            print(f"    z>={T:.1f}: {n}/{len(zs)} ({100*n/len(zs):.0f}%)")
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
