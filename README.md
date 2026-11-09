# Bit-Flip Fault Injection in an FP16 MNIST MLP

## Research questions
RQ1 (required). Under randomly chosen persistent single-bit FP16 weight faults, how do the
text encoder, UNet blocks and VAE decoder differ in their rates of human-validated semantic
change and visual corruption?
RQ2 (extension). With the same model and sampler, do more denoising steps reduce the effect
of a transient fault injected at the same noise level and size?
RQ3 (scale check). Do conclusions from a small model hold for full Stable Diffusion 1.5?

This repository contains the pilot on an MNIST MLP, which builds and tests the fault
injection method for these questions.

## Master seed
`20261104`. All randomness is derived from it with NumPy's `SeedSequence`, one stream per
purpose: spawn key 0 for training, 1 for the single-weight walk-through, 2 and 3 for
sampling the two weight matrices.

## Library versions
Python on Google Colab with a Tesla T4 GPU:
```
torch==2.11.0+cu130
torchvision==0.26.0+cu130
numpy==2.4.6
pandas==2.2.3
matplotlib==3.10.0
```
These are pinned in `requirements.txt`.

## Reproduce the tables
On Google Colab with a GPU runtime (Runtime → Change runtime type → T4 GPU):
```bash
git clone https://github.com/AryanKpr/AIC-Research.git && cd AIC-Research && pip install -r requirements.txt && python src/train.py && python src/test_flip.py && python src/step4.py && python src/campaign.py run1 && python src/analyze.py run1
```
This produces the 16-bit walk-through table (`logs/step4.csv`), the campaign results
(`logs/campaign_run1.csv`) and the per-bit summary with Wilson intervals
(`logs/summary_run1.csv`). The results were produced at commit
`7f44b71fe95e513da97a68f9e002175ce532a855`.


## Team
Aryan Kapoor

Jai Totla
