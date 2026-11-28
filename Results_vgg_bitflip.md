# Bit-flip faults in a small VGG: results by bit and by layer

**Question.** When a single bit flips in a stored FP16 weight of a small convolutional network,
how does the outcome depend on the bit position and on the layer, and does it differ between
Fashion-MNIST and CIFAR-10?

**Short answer.** Sign and exponent flips almost always change predictions (61–100% of faults,
depending on bit and dataset). Exponent bit 14 changes predictions in essentially every fault, but
unlike the week 1 MLP it rarely produces inf or NaN (3–4% vs 26%). Mantissa flips often change a
handful of predictions, but only by tipping a few test images whose top two scores are nearly tied;
counting only faults that change 5 or more predictions, mantissa bits 0–3 are harmless. The first
convolution is by far the most fragile layer and the last convolution the most robust. CIFAR-10 is
more sensitive than Fashion-MNIST at almost every bit.

---

## 1. Setup

| | Fashion-MNIST | CIFAR-10 |
|---|---|---|
| Network | small VGG (spec table), 288,170 parameters | small VGG, 288,746 parameters |
| Weights file SHA-256 | `41bb17a5…765d` | `a60d257c…a288` |
| Fault-free FP16 test accuracy | TODO (from `logs/train_vgg_fmnist.json`) | 89.37% |
| GPU for the campaign | Tesla T4 | Tesla T4 |

Each network was trained once and frozen. Every campaign loads it through `load_trained()`, which
refuses a file whose SHA-256 does not match the training record.

## 2. Fault model

- **What:** a single bit flip in a stored FP16 weight, applied in place to the tensor the network
  uses, kept for one evaluation of all 10,000 test images, then removed by writing the saved bit
  pattern back. A SHA-256 fingerprint of the layer is checked after every restore; the run stops on
  any mismatch.
- **Where:** the weights of the six convolutions (conv1–conv6) and the final linear layer (fc).
  Batch normalization parameters and statistics were not faulted.
- **Which elements:** 100 distinct elements per layer, drawn without replacement, each layer from
  its own random stream derived from the master seed 20261104 (role 5, keyed by architecture,
  dataset and layer name). The drawn indices are saved in `configs/`.
- **Which bits:** all 16 (15 sign, 14–10 exponent, 9–0 mantissa).
- **How many:** 7 layers × 100 elements × 16 bits = **11,200 faults per dataset**. Every layer
  contributes equally, so pooled results describe a "typical layer", not the whole network's risk.
- **Evaluation:** all 10,000 test images in fixed batches of 1,000, with deterministic GPU settings.
- **Classification**, in this order: *non-finite* if any output is inf or NaN; *SDC* if all outputs
  are finite but at least one prediction differs from the fault-free network; otherwise *masked*.
  For each SDC we also record **how many** of the 10,000 predictions changed.

## 3. Checks that the measurement is sound

| Check | Result |
|---|---|
| A flip changes the real network (sanity check on conv6, bit 14) | passed |
| Every restore exact (SHA-256 after each of 22,400 faults) | passed |
| Fault-free network gives identical outputs over 5 evaluations | identical, both datasets |
| Full CIFAR-10 campaign run twice on the same GPU type | **identical files**, SHA-256 `484f00d5…afbde6` |

The two CIFAR-10 runs recorded different commits (`ecd7cdc` and `1395b56`); the only file changed
between them was the analysis script, which does not affect the campaign.

## 4. Results by bit (all layers pooled, n = 700 per bit)

Percentages with 95% Wilson intervals.

| Bit | Field | CIFAR-10 SDC | CIFAR-10 non-finite | Fashion-MNIST SDC | Fashion-MNIST non-finite |
|---|---|---|---|---|---|
| 15 | sign | 91.6 [89.3, 93.4] | 0.0 [0.0, 0.5] | 76.1 [72.8, 79.2] | 0.0 [0.0, 0.5] |
| 14 | exponent | 96.0 [94.3, 97.2] | 4.0 [2.8, 5.7] | 96.6 [94.9, 97.7] | 3.0 [2.0, 4.5] |
| 13 | exponent | 94.3 [92.3, 95.8] | 0.0 [0.0, 0.5] | 81.4 [78.4, 84.1] | 0.0 [0.0, 0.5] |
| 12 | exponent | 95.0 [93.1, 96.4] | 0.0 [0.0, 0.5] | 82.1 [79.1, 84.8] | 0.0 [0.0, 0.5] |
| 11 | exponent | 91.6 [89.3, 93.4] | 0.0 [0.0, 0.5] | 74.3 [70.9, 77.4] | 0.0 [0.0, 0.5] |
| 10 | exponent | 87.7 [85.1, 89.9] | 0.0 [0.0, 0.5] | 61.1 [57.5, 64.7] | 0.0 [0.0, 0.5] |
| 9 | mantissa | 80.6 [77.5, 83.3] | 0.0 [0.0, 0.5] | 49.6 [45.9, 53.3] | 0.0 [0.0, 0.5] |
| 8 | mantissa | 74.4 [71.1, 77.5] | 0.0 [0.0, 0.5] | 40.7 [37.1, 44.4] | 0.0 [0.0, 0.5] |
| 7 | mantissa | 67.1 [63.6, 70.5] | 0.0 [0.0, 0.5] | 28.7 [25.5, 32.2] | 0.0 [0.0, 0.5] |
| 6 | mantissa | 62.9 [59.2, 66.4] | 0.0 [0.0, 0.5] | 19.9 [17.1, 23.0] | 0.0 [0.0, 0.5] |
| 5 | mantissa | 56.3 [52.6, 59.9] | 0.0 [0.0, 0.5] | 13.0 [10.7, 15.7] | 0.0 [0.0, 0.5] |
| 4 | mantissa | 50.4 [46.7, 54.1] | 0.0 [0.0, 0.5] | 9.6 [7.6, 12.0] | 0.0 [0.0, 0.5] |
| 3 | mantissa | 44.6 [40.9, 48.3] | 0.0 [0.0, 0.5] | 5.9 [4.3, 7.8] | 0.0 [0.0, 0.5] |
| 2 | mantissa | 39.9 [36.3, 43.5] | 0.0 [0.0, 0.5] | 6.1 [4.6, 8.2] | 0.0 [0.0, 0.5] |
| 1 | mantissa | 35.0 [31.6, 38.6] | 0.0 [0.0, 0.5] | 3.7 [2.5, 5.4] | 0.0 [0.0, 0.5] |
| 0 | mantissa | 29.0 [25.8, 32.5] | 0.0 [0.0, 0.5] | 3.4 [2.3, 5.1] | 0.0 [0.0, 0.5] |

![VGG per bit, Fashion-MNIST vs CIFAR-10](../figures/vgg_compare_run1.png)

### How severe are the SDCs?

"SDC" counts one changed prediction the same as a collapse. Splitting by the number of changed
predictions separates the two:

| Bit | Field | CIFAR-10 any change | CIFAR-10 ≥ 5 changed | Fashion-MNIST any change | Fashion-MNIST ≥ 5 changed |
|---|---|---|---|---|---|
| 15 | sign | 91.6 | 64.6 [61.0, 68.0] | 76.1 | 44.4 [40.8, 48.1] |
| 14 | exponent | 100.0 | 98.3 [97.0, 99.0] | 99.6 | 96.6 [94.9, 97.7] |
| 13 | exponent | 94.3 | 61.4 [57.8, 65.0] | 81.4 | 46.6 [42.9, 50.3] |
| 12 | exponent | 95.0 | 79.1 [76.0, 82.0] | 82.1 | 69.0 [65.5, 72.3] |
| 11 | exponent | 91.6 | 59.6 [55.9, 63.1] | 74.3 | 40.6 [37.0, 44.3] |
| 10 | exponent | 87.7 | 43.6 [39.9, 47.3] | 61.1 | 26.7 [23.6, 30.1] |
| 9 | mantissa | 80.6 | 28.0 [24.8, 31.4] | 49.6 | 18.1 [15.5, 21.2] |
| 8 | mantissa | 74.4 | 17.1 [14.5, 20.1] | 40.7 | 11.4 [9.3, 14.0] |
| 7 | mantissa | 67.1 | 8.7 [6.8, 11.0] | 28.7 | 6.1 [4.6, 8.2] |
| 6 | mantissa | 62.9 | 4.6 [3.3, 6.4] | 19.9 | 3.1 [2.1, 4.7] |
| 5 | mantissa | 56.3 | 2.1 [1.3, 3.5] | 13.0 | 1.9 [1.1, 3.2] |
| 4 | mantissa | 50.4 | 0.3 [0.1, 1.0] | 9.6 | 0.4 [0.1, 1.3] |
| 3 | mantissa | 44.6 | 0.0 [0.0, 0.5] | 5.9 | 0.1 [0.0, 0.8] |
| 2–0 | mantissa | 29.0–39.9 | 0.0 [0.0, 0.5] each | 3.4–6.1 | 0.0 [0.0, 0.5] each |

("Any change" includes non-finite faults; the ≥ 5 column counts non-finite faults as failures too.)

The fault-free networks have a few borderline test images whose top two class scores are less
than 0.01 apart: 14 on CIFAR-10 and 9 on Fashion-MNIST. Almost every SDC from bits 0–3 changed only
1–3 predictions, consistent with a tiny weight change tipping one of these borderline images. The
threshold of 5 is a first choice, not a fixed rule.

## 5. Results by layer

Share of faults that failed (SDC or non-finite), by layer and bit field, in %
(n = 100 for sign, 500 for exponent, 1,000 for mantissa per layer):

| Layer | CIFAR-10 sign | CIFAR-10 exponent | CIFAR-10 mantissa | Fashion-MNIST sign | Fashion-MNIST exponent | Fashion-MNIST mantissa |
|---|---|---|---|---|---|---|
| conv1 | 100.0 | 100.0 | 97.5 | 100.0 | 99.0 | 60.7 |
| conv2 | 100.0 | 99.8 | 83.8 | 82.0 | 83.2 | 13.7 |
| conv3 | 100.0 | 99.8 | 77.9 | 89.0 | 88.8 | 16.3 |
| conv4 | 95.0 | 97.0 | 55.1 | 88.0 | 88.6 | 15.3 |
| conv5 | 97.0 | 96.8 | 32.4 | 77.0 | 79.8 | 8.9 |
| conv6 | 52.0 | 65.6 | 3.4 | 32.0 | 51.2 | 0.3 |
| fc | 97.0 | 97.0 | 28.0 | 65.0 | 67.4 | 11.2 |

![CIFAR-10 heatmap](../figures/vgg_cifar10_run1_heatmap.png)
![Fashion-MNIST heatmap](../figures/vgg_fmnist_run1_heatmap.png)

Per-layer plots with intervals: `figures/vgg_cifar10_run1_layers.png`,
`figures/vgg_fmnist_run1_layers.png`. Per-layer, per-bit tables:
`logs/summary_vgg_<dataset>_run1_layer_bit.csv`.

## 6. Findings

**1. Exponent bit 14 fails almost every time, but mostly silently.** It changed predictions in
100% (CIFAR-10) and 99.6% (Fashion-MNIST) of faults, and 97–98% changed 5 or more. Only 4.0% and
3.0% gave inf or NaN, against 25.7% in the week 1 MLP. *Likely reason (to verify with a weight
histogram):* weight decay keeps VGG weights small, and a bit-14 flip only produces inf or NaN
directly when |w| ≥ 1; smaller weights become large but finite, and batch normalization and
pooling pass them on as wrong values rather than overflow.

**2. Sign and lower exponent flips are much more damaging than in the MLP.** On CIFAR-10, bits 15
and 13–10 changed predictions in 88–95% of faults (MLP: 50% for the sign bit, 38–73% for bits 13–10). Under the ≥ 5 threshold, bit 12 is again worse than
bits 13 and 11 (79% vs 61% and 60% on CIFAR-10), the same ordering seen in the MLP; *likely reason:*
for typical weights bit 12 is 0 (a flip multiplies by 16) while bit 13 is often 1 (a flip divides
by 256).

**3. Mantissa flips cause small, borderline changes.** Under "any prediction changed", mantissa
SDC stays high (29% at bit 0 on CIFAR-10), but these faults change only 1–3 predictions. Counting
faults that change 5 or more, bits 0–3 caused none on CIFAR-10 and one on Fashion-MNIST, matching
the MLP result that the lowest mantissa bits are harmless.

**4. The first convolution is the most fragile layer; the last convolution the most robust.** On
CIFAR-10, the mantissa failure rate falls from 97.5% in conv1 to 3.4% in conv6, and the sign
failure rate from 100% to 52%. *Likely reason:* fan-in. Each conv1 output sums only 27 inputs
(9 on Fashion-MNIST), so one weight is a large share of it, and conv1 runs at full image
resolution; each conv6 output sums 1,152 inputs. The fc layer, with 128 inputs feeding the class
scores directly, sits in between.

**5. CIFAR-10 is more sensitive than Fashion-MNIST.** CIFAR-10 has the higher failure rate at
nearly every bit and every layer (for example bit 9: 81% vs 50%). This fits the harder task having
narrower decision margins, and the larger number of borderline test images (14 vs 9).

## 7. Limitations

- One trained network per dataset; results may vary between training runs.
- Equal sampling per layer: pooled rates describe a typical layer, not risk to the whole network,
  where conv5 and conv6 hold most weights.
- n = 100 per layer and bit gives intervals of up to about ±10 percentage points per cell.
- Batch normalization parameters and statistics were not faulted.
- "Any prediction changed" is very sensitive; the ≥ 5 threshold is a first choice.
- The weight-size and fan-in explanations are hypotheses, not yet tested.

## 8. Open questions

1. Which SDC definition to report as primary: any change, or a threshold such as ≥ 5 or > 0.1% of
   the test set?
2. Should batch normalization parameters and running statistics be faulted in a follow-up?
3. Should the per-layer sample size be raised (for example to 150) for tighter per-layer intervals?

## 9. Reproduce

```bash
python src/check_cnn.py                         # verifies both weight files by SHA-256
python src/campaign_cnn.py vgg cifar10 run1
python src/campaign_cnn.py vgg fmnist run1
python src/analyze_cnn.py vgg cifar10 run1
python src/analyze_cnn.py vgg fmnist run1
```

Raw results: `logs/campaign_vgg_<dataset>_run1.csv` (one row per fault, including the number of
changed predictions). Run metadata: `logs/campaign_vgg_<dataset>_run1_meta.json`.
