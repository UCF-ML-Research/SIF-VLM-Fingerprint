# SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models

This is the official implementation of the paper [SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models](https://arxiv.org/abs/2604.17041), which has been accepted to CVPR 2026.

### TL;DR

Existing fingerprinting methods for Large Vision-Language Models (LVLMs) rely on semantically abnormal queries or out-of-distribution responses. **SIF** identified that this could be easily exploited by adversaries to evade fingerprint verification. To address this, SIF distills text-generation watermark signals into the input image, so the model naturally produces semantically coherent responses that carry a hidden, statistically verifiable watermark — enabling practical and robust copyright verification without modifying model parameters.

![Overview of SIF fingerprint](diagram/diagram.png)

### Setup

```bash
bash install.sh
```

### Code Structure

```
sif/                              # SIF method (ours)
├── sif.py, detect.py             # Generate triggers, verify on source/suspect/unrelated
├── utils.py                      # Shared utilities
├── generate.sh, detect.sh        # bash {generate|detect}.sh {llava|qwen}
└── watermarks/kgw/               # KGW text watermark scheme

fingerprint/                      # Baselines (7 methods)
├── generate.py, detect.py        # Unified generation & verification CLI
├── generate.sh, detect.sh        # bash {generate|detect}.sh {method} {llava|qwen}
└── core/                         # pla, ordinary, rna, cropa, difgsm, proflingo/, instruction_fingerprint/

stealthiness/                     # Stealthiness evaluation (PPL, divergence, GPT judge)
imagenet/                         # Source images
```

### SIF

```bash
bash sif/generate.sh llava     # Generate fingerprint triggers (SAFD + RFO)
bash sif/detect.sh llava            # Verify across the LLaVA suspect/unrelated roster
```

### Baselines

| Method | Type | Command |
|---|---|---|
| PLA | Image (bilevel PGD) | `bash generate.sh pla llava` |
| Ordinary | Image (vanilla PGD) | `bash generate.sh ordinary llava` |
| RNA | Image (noise on weights) | `bash generate.sh rna llava` |
| CroPA | Image (text emb perturb) | `bash generate.sh cropa llava` |
| DI²-FGSM | Image (input diversity) | `bash generate.sh difgsm llava` |
| ProFLingo | Text (suffix optim) | `bash generate.sh proflingo llava` |
| Instruction FP | Training (LoRA) | `bash generate.sh instruction_fingerprint llava` |

Replace `llava` with `qwen` for Qwen2.5-VL-7B. All commands run from `fingerprint/`. Verify any method with `bash detect.sh {method} {llava|qwen}`.

### Stealthiness Evaluation

See [`stealthiness/`](stealthiness/) for details.

### Acknowledgements

Our watermark implementation is based on [watermark-learnability](https://github.com/chenchenygu/watermark-learnability). Our visual adversarial optimization is inspired by [Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models). We thank the authors for their open-source contributions.

### Citation

```bibtex
@article{zhao2026sif,
  title={SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models},
  author={Zhao, Yifei and Lou, Qian and Zheng, Mengxin},
  journal={arXiv preprint arXiv:2604.17041},
  year={2026}
}
```
