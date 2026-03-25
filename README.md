# SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models

This is the official implementation of the paper [SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models], which has been accepted to CVPR 2026.

### TL;DR

Existing fingerprinting methods for Large Vision-Language Models (LVLMs) rely on semantically abnormal queries or out-of-distribution responses. **SIF** identified that this could be easily exploited by adversaries to evade fingerprint verification. To address this, SIF distills text-generation watermark signals into the input image, so the model naturally produces semantically coherent responses that carry a hidden, statistically verifiable watermark — enabling practical and robust copyright verification without modifying model parameters.

![Overview of SIF fingerprint](diagram/diagram.png)

### Setup

To install the necessary packages, first create a conda environment.
```
conda create -n <env_name> python=3.11
conda activate <env_name>
```
Then, install the required packages with 
```
pip install -r requirements.txt
```


### Code Structure

```
sif/
├── sif.py               # SIF fingerprint generation (Qwen2.5-VL)
├── detect.py            # Fingerprint detection via watermark z-score test
├── generate.sh          # Script to generate fingerprint trigger images
├── fingerprint.sh       # Script to run fingerprint detection
├── sif_fingerprint/     # Pre-generated fingerprint trigger images
└── watermarks/kgw/      # KGW text watermark scheme (watermark logits processor & detector)
stealthiness/
├── judge.py             # GPT-4.1-based stealthiness judge
└── judge.sh             # Stealthiness evaluation script
pla/                     # Parameter Learning Attack (baseline)
```

Key components:
- **Differentiable vision preprocessing**: `DiffQwen2VLFast` (in `sif/sif.py`) and `DiffLLaVAPreprocess` (in `sif/SAFD.py`) — differentiable image-to-patch-token pipelines that enable gradient-based PGD optimization on the input image.
- **PGD optimization**: In `sif/sif.py` and `sif/SAFD.py` — optimizes trigger images with watermark alignment loss + semantic alignment loss under an L-inf perturbation budget (`eps=16/255`, `alpha=1/255`, 1000 steps).
- **Watermark scheme**: `sif/watermarks/kgw/watermark_processor.py` — implements the text watermark.

### SIF

Generate fingerprint trigger images:
```bash
bash sif/generate.sh
```

Pre-generated fingerprints are stored in `sif/sif_fingerprint/`.

Run fingerprint detection on a suspect model:
```bash
bash sif/fingerprint.sh
```

### Stealthiness Evaluation

Uses a VLM judge to expose the semantically in-distribution and stealthy nature of SIF fingerprints. Configure your OpenAI API key in `stealthiness/judge.sh`, then run:

```bash
bash stealthiness/judge.sh
```

### Parameter Learning Attack

The Parameter Learning Attack (PLA) baseline is implemented in the `pla/` directory.

```bash
bash pla/fingerprint.sh
```

### Acknowledgements

Our watermark implementation is based on [watermark-learnability](https://github.com/chenchenygu/watermark-learnability). Our visual adversarial optimization is inspired by [Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models). We thank the authors for their open-source contributions.

### Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{zhao2026sif,
  title={SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models},
  author={Zhao, Yifei and Lou, Qian and Zheng, Mengxin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
