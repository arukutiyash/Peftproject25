# PEFT-Privacy-Eval : A Unified Framework for Attacking & Defending Parameter-Efficient Fine-Tuning  

Comprehensive codebase accompanying the research on **privacy leakage in PEFT models** (Adapters, Prefix-Tuning, Bias-Tuning, LoRA) inside a Federated-Learning (FL) pipeline.

> Re-implements the gradient-inversion attack from  
> *“Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning”, CVPR 2025*  
> and extends it with multiple PEFT methods, multi-round attacks, and defenses.

---

## 🌳 Directory Tree

attacks/ # all attack implementations
defenses/ # individual & combined defense algorithms
evaluation/ # metrics, heat-maps, statistical analysis
federated/ # minimal FL simulator (clients + server)
main/ # runnable experiment entry-points
models/ # ViT backbone + PEFT modules (adapters / prefix / bias / lora)
utils/ # data loading, patch recovery, visualisation, misc helpers
requirements.txt

### Key Files

| Path | Purpose |
|------|---------|
| `models/vit_model.py` | Vision Transformer backbone |
| `models/peft_*.py` | Adapter / Prefix / Bias / LoRA layers |
| `attacks/adapter_attack.py` | CVPR-2025 adapter attack |
| `attacks/multi_round_attack.py` | Attack spanning several FL rounds |
| `defenses/differential_privacy.py` | DP-SGD noise on gradients |
| `utils/metrics.py` | PSNR, SSIM, LPIPS, recovery-rate |
| `main/run_attack_experiments.py` | One-shot script to benchmark attacks |
| `main/run_defense_experiments.py` | Evaluate chosen defenses |
| `main/config.py` | Central place for experiment settings |

---
1. clone & enter
git clone 
cd peft-privacy-eval

2. python env (3.8+)
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

3. install deps
pip install -r requirements.txt # ~90 s on CPU

4. run a demo attack (Adapters, CIFAR-100, 1 FL round)
python main/run_attack_experiments.py
--peft-method adapter
--dataset cifar100
--batch-size 32
--adapter-dim 64

→ outputs/recon_grid.png (≈ Fig. 4 of the paper)
text

---

## 🛠️ Command-Line Cheatsheet

| Flag | Meaning | Typical |
|------|---------|---------|
| `--peft-method {adapter,prefix,bias,lora}` | choose PEFT layer | adapter |
| `--dataset {cifar10,cifar100,tinyimagenet,imagenet}` | auto-download except ImageNet | cifar100 |
| `--batch-size` | images per client | 32 |
| `--rounds` | FL rounds (≥ 2 enables multi-round attack) | 1 |
| `--adapter-dim / --prefix-length / ...` | method-specific dim | 64 |
| `--defense {none,mixup,instahide,dp,grad_prune}` | add defense | none |
| `--noise-multiplier` | σ for DP-SGD | 1.0 |
| `--save-dir` | where to store logs & grids | outputs/run-DATE |

Full list: `python main/run_attack_experiments.py -h`.

---

## 🚀 Typical Workflows

### 1 ▪ Multi-Round Attack with Small Adapters  
(reproduces Fig. 7)

python main/run_attack_experiments.py
--peft-method adapter
--adapter-dim 8
--rounds 5
--dataset cifar100

text

### 2 ▪ Evaluate Differential-Privacy Defense

python main/run_defense_experiments.py
--defense dp
--noise-multiplier 1.0
--peft-method adapter
--dataset tinyimagenet

text

### 3 ▪ Generate Complete Result Tables / Heat-maps

python main/generate_results.py
--output-dir results/
--datasets cifar10 cifar100 tinyimagenet
--peft-methods adapter prefix bias lora
--include-defenses
python evaluation/heatmap_generator.py --metric-type psnr

text

---

## 📊 Metrics Produced

* **Patch-Recovery Rate** ( % of patches exactly reconstructed )  
* **PSNR / SSIM / LPIPS** between ground-truth & reconstructed images  
* **Accuracy Loss** when defenses are enabled  
* **Statistical Significance** via paired t-test (evaluation/statistical_analysis.py)

All numbers are dumped as JSON and can be compiled into LaTeX tables via `evaluation/report_generator.py`.

---

## 🔬 Extend the Framework

* **New PEFT method** → implement in `models/peft_xyz.py`, register in `model_factory.py`.  
* **New attack** → subclass `attacks/attack_base.AttackBase`, plug into `run_attack_experiments.py`.  
* **New defense** → subclass `defenses/defense_base.DefenseBase`, add to `defense_factory.py`.

Unit tests are welcome (see `tests/` placeholder).

---

## 📑 Citation

@inproceedings{sami2025peftleak,
title = {Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning},
author = {Hasin Us Sami and Swapneel Sen and Amit K. Roy-Chowdhury and
Srikanth V. Krishnamurthy}

---

## ⚠️ Disclaimer

This project is **research software**.  
Run the attacks **only** on data and systems you own or have explicit permission to test.

---
Contact:
Have any queries! Feel free to mail me at Yashwanthsaiarukuti@my.unt.edu 


