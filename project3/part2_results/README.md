# Part 2: weak (point) supervision on PH2. Material for the report

Everything needed to write Part 2. Numbers come from `project3/results/*.json` (3 seeds per configuration).
The code is in `project3/` (`lib/clicks.py`, `lib/losses.py`, `train.py`, `jobscripts/weak.sh`).

Files in this folder:

| File | What it is |
|---|---|
| `clicks.png` | The three sampling strategies on the same 3 training images (10+10 clicks). **Main figure.** |
| `predictions.png` | Test predictions for full masks vs. 10+10 clicks per strategy. Optional, but it explains the boundary result. |
| `clicks_ablation_test.png` | Test Dice vs. number of clicks per strategy, full masks as a dashed line. Optional plot of the table. |
| `table_test.tex`, `table_val.tex` | Results tables in LaTeX (need `\usepackage{booktabs}`) |
| `table_test.md`, `table_val.md` | The same tables as Markdown |

---

## 1. Setup (identical to Part 1 except for the labels)

- **Data:** PH2, the same 140/30/30 train/val/test split as Part 1 (same fixed shuffle).
- **Input:** images resized to 256×256. No augmentation.
- **Training:** Adam, learning rate 1e-3, batch size 6, 50 epochs. The model after the last epoch is evaluated; validation is not used for model selection.
- **Weak labels in training only:** only the training set uses clicks. Validation and test are always scored against the **full** masks.
- **Metrics:** threshold p = 0.5. The confusion matrix is pooled over the whole split, then Dice, IoU, accuracy, sensitivity, specificity and precision are computed from it.
- **Seeds:** every configuration is run with 3 seeds (0, 1, 2). The seed sets the network initialisation, the batch order **and** the clicks, so the reported std covers annotation variability as well as training noise.
- **Baseline:** the fully supervised U-Net was retrained with the same 3 seeds, which gives a fair baseline: Dice 0.910 ± 0.016.

## 2. Architecture

The same U-Net as Part 1, unchanged:

- **Encoder:** 4 levels. Each level is a 3×3 conv (64 channels, padding 1), then ReLU, then 2×2 max-pool. Spatial size goes 256 → 128 → 64 → 32 → 16.
- **Bottleneck:** one 3×3 conv with 64 channels, then ReLU, at 16×16.
- **Decoder:** 4 levels. Each level upsamples ×2 (nearest neighbour), concatenates the encoder feature map at the same resolution (skip connection, 64 + 64 = 128 channels), then applies a 3×3 conv to 64 channels and ReLU. Spatial size goes 16 → 32 → 64 → 128 → 256.
- **Output:** the last decoder conv maps to **1 channel** (a logit per pixel), with no activation; the sigmoid is applied inside the loss.
- **Size:** 372,033 parameters. No batch norm. One conv per level, so lighter than the original U-Net.

Feature-map sizes (C × H × W):

```
input            3 × 256 × 256
enc0 conv       64 × 256 × 256  -> pool  64 × 128 × 128
enc1 conv       64 × 128 × 128  -> pool  64 ×  64 ×  64
enc2 conv       64 ×  64 ×  64  -> pool  64 ×  32 ×  32
enc3 conv       64 ×  32 ×  32  -> pool  64 ×  16 ×  16
bottleneck      64 ×  16 ×  16
up + skip enc3 128 ×  32 ×  32  -> conv  64 ×  32 ×  32
up + skip enc2 128 ×  64 ×  64  -> conv  64 ×  64 ×  64
up + skip enc1 128 × 128 × 128  -> conv  64 × 128 × 128
up + skip enc0 128 × 256 × 256  -> conv   1 × 256 × 256  (logits)
```

## 3. Loss function (point-supervised BCE)

The network still predicts a full 256×256 map, but the loss is computed **only at the clicked pixels**.
All other pixels are "unknown" and get no gradient. Let $P$ be the positive clicks and $N$ the negative
clicks in a batch, and $p_i = \sigma(x_i)$ the predicted lesion probability at pixel $i$:

$$
\mathcal{L}_{\text{point}} = -\frac{1}{|P|}\sum_{i\in P}\log p_i \;-\; \frac{1}{|N|}\sum_{i\in N}\log(1-p_i)
$$

- **It is ordinary binary cross-entropy restricted to the labelled pixels.** This is the point-supervision term of Bearman et al. (ECCV 2016, eq. 3 in the lecture). Their image-level term is dropped because every PH2 image contains exactly one lesion, so it carries no information.
- **The two classes are averaged separately,** so positives and negatives weigh equally even if their counts differ.
- **Why a few points are enough:** the conv filters are shared across all pixels, so what is learnt at a click applies everywhere. Neighbouring pixels share most of their receptive field, so predictions spread from the clicks into coherent regions.
- **Why negatives are needed:** with positive clicks only, "predict everything as lesion" would give zero loss.
- **Implementation:** `PointBCELoss` in `lib/losses.py`. The dataloader (`PH2Clicks` in `lib/dataset/PH2Dataset.py`) returns, instead of the mask, a 2-channel target: the click label and a map marking which pixels were clicked.

## 4. Click sampling strategies (see `clicks.png`)

Shared procedure for all strategies:

1. **Resolution:** clicks are sampled from the true mask at the training resolution (256×256), so all distances below are in 256-pixel units (1 px ≈ 3 px in the original image).
2. **Distance to the edge:** for every pixel the distance to the lesion boundary is computed:
   - $d_{in}$ for pixels inside the lesion, the distance to the nearest background pixel;
   - $d_{out}$ for pixels outside, the distance to the nearest lesion pixel.

   It is computed by repeated 3×3 erosion (chessboard distance); pixels at the edge have $d = 1$.
3. **Probability maps:** each strategy assigns a weight to every pixel, one map for positives and one for negatives.
4. **Drawing clicks:** $N$ positives and $N$ negatives are drawn **without replacement**, with probability proportional to the weight. If a weight map is empty, the strategy falls back to random.
5. **Fixed clicks:** clicks are sampled **once per image** and kept fixed for all epochs, simulating an annotator who clicks each image once. Resampling every epoch would let the network see thousands of different labelled pixels per image, close to full supervision.

Weights per strategy ($y_i$ = true label of pixel $i$):

| Strategy | Positive clicks: weight $w^+_i$ | Negative clicks: weight $w^-_i$ |
|---|---|---|
| **random** | $\mathbb{1}[y_i=1]$ | $\mathbb{1}[y_i=0]$ |
| **interior** | $\max(d_{in}(i)-3,\,0)$ | $\mathbb{1}[3 < d_{out}(i) \le 40]$ |
| **boundary** | $\mathbb{1}[1 \le d_{in}(i) \le 10]$ | $\mathbb{1}[1 \le d_{out}(i) \le 10]$ |

**Random:** uniform sampling inside the mask (positives) and outside it (negatives). This is the baseline strategy the assignment suggests.

**Interior (annotator-like):** simulates how a person would click.
- **Positives:** the weight grows linearly with the distance to the edge, so clicks are pulled towards the centre of the lesion, and nothing lands within 3 px of the edge, where a person would be unsure. For a round lesion of radius $R$, the average click lies at $R/2$ from the centre, against $2R/3$ for uniform sampling.
- **Negatives:** uniform over a ring of skin 3–40 px outside the lesion (about 16% of the image width). A person marks where the lesion stops, not the dark corners outside the dermoscope.

**Boundary (hard examples):** every click lies within 10 px of the edge, positives just inside and negatives just outside, uniform within those bands.
- This puts all the information where the decision is hardest.
- It is unrealistic, because people are least reliable exactly there.
- It is included to test whether informative clicks beat well-spread ones.

**Limitations to mention:**
- The thresholds (3, 40 and 10 px) are hand-picked, not tuned.
- The clicks never carry a wrong label, so the strategies simulate *where* a person clicks, not human mistakes.
- One training image is about 98% lesion; there, negatives can only land in the dark corners.

## 5. Results

Test set (30 images), mean ± std over 3 seeds. Clicks are per class per training image (N positive + N negative).
The validation table (`table_val.*`) shows the same pattern.

| Supervision | Clicks (pos+neg) | Dice | IoU | Acc. | Sens. | Spec. | Prec. |
|---|---|---|---|---|---|---|---|
| Full masks | -- | 0.910 ± 0.016 | 0.836 ± 0.027 | 0.948 ± 0.011 | 0.942 ± 0.022 | 0.951 ± 0.023 | 0.883 ± 0.046 |
| random | 1+1 | 0.786 ± 0.060 | 0.651 ± 0.080 | 0.893 ± 0.021 | 0.731 ± 0.130 | 0.955 ± 0.040 | 0.882 ± 0.082 |
| random | 3+3 | 0.868 ± 0.007 | 0.766 ± 0.012 | 0.926 ± 0.003 | 0.885 ± 0.025 | 0.941 ± 0.007 | 0.852 ± 0.011 |
| random | 10+10 | 0.890 ± 0.009 | 0.801 ± 0.015 | 0.937 ± 0.007 | 0.916 ± 0.024 | 0.945 ± 0.016 | 0.866 ± 0.033 |
| random | 30+30 | 0.907 ± 0.006 | 0.830 ± 0.009 | 0.948 ± 0.004 | 0.920 ± 0.026 | 0.959 ± 0.013 | 0.896 ± 0.027 |
| interior | 1+1 | 0.822 ± 0.016 | 0.698 ± 0.022 | 0.906 ± 0.007 | 0.790 ± 0.028 | 0.950 ± 0.002 | 0.858 ± 0.004 |
| interior | 3+3 | 0.869 ± 0.005 | 0.768 ± 0.008 | 0.929 ± 0.003 | 0.850 ± 0.019 | 0.959 ± 0.009 | 0.889 ± 0.021 |
| interior | 10+10 | 0.888 ± 0.011 | 0.799 ± 0.019 | 0.939 ± 0.006 | 0.874 ± 0.028 | 0.964 ± 0.012 | 0.905 ± 0.028 |
| interior | 30+30 | 0.901 ± 0.005 | 0.820 ± 0.009 | 0.946 ± 0.004 | 0.901 ± 0.024 | 0.962 ± 0.014 | 0.903 ± 0.031 |
| boundary | 1+1 | 0.644 ± 0.047 | 0.477 ± 0.052 | 0.777 ± 0.026 | 0.734 ± 0.073 | 0.794 ± 0.015 | 0.574 ± 0.034 |
| boundary | 3+3 | 0.708 ± 0.046 | 0.550 ± 0.056 | 0.821 ± 0.036 | 0.783 ± 0.067 | 0.835 ± 0.058 | 0.654 ± 0.078 |
| boundary | 10+10 | 0.753 ± 0.017 | 0.604 ± 0.021 | 0.848 ± 0.008 | 0.843 ± 0.034 | 0.849 ± 0.009 | 0.680 ± 0.011 |
| boundary | 30+30 | 0.773 ± 0.036 | 0.631 ± 0.047 | 0.855 ± 0.036 | 0.881 ± 0.030 | 0.845 ± 0.059 | 0.694 ± 0.071 |

### Main observations (to build the discussion on)

1. **How many clicks match full supervision:**
   - **30+30 random clicks** (0.907 ± 0.006) are indistinguishable from full masks (0.910 ± 0.016).
   - **10+10** is about 0.02 Dice lower, roughly one standard deviation of the baseline.
   - Most of the gain comes early: **3+3 already reaches about 0.87**.
2. **Interior vs. random:** interior is clearly better with a single click per class (0.822 ± 0.016 vs 0.786 ± 0.060) and far more stable across seeds. A single random positive can land on the faint rim of the lesion, while an interior click always marks clearly lesion-like tissue. From 3+3 upwards the two are equal within noise.
3. **Boundary clicks fail,** and more clicks barely help (0.77 at 30+30):
   - `predictions.png` shows that the boundary models trace the lesion outline well but mark distant skin and the dark corners outside the dermoscope as lesion.
   - This shows up as low specificity (about 0.85 vs about 0.95) and precision.
   - The cause: their negatives never lie more than 10 px from the lesion, so the network never sees most of the background.
   - Interior has a milder version of this (small spurious blobs), since its negatives stop at 40 px.
   - The lesson: **negatives must cover the variety of the background; positives benefit from being confidently inside the object.**
4. **The test set is small (30 images):** differences under about 0.01–0.02 Dice are within seed noise. One test image with a small, faint lesion is hard for every model, including full masks (Dice ≈ 0.39 there), and accounts for much of the spread.

### How to regenerate
```bash
cd project3
mkdir -p logs && bsub -app c02516_1g.10gb < jobscripts/weak.sh   # 39 runs, ~20 min
python summarize.py              # table + clicks_ablation_test.png
python show_clicks.py --clicks 10 --images 3
python show_predictions.py --clicks 10 --seed 0 --images 4
```
