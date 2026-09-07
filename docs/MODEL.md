# Model Architecture & Methodology Deep Dive

This document explains the mathematical, visual, and operational rationales behind the modeling decisions in this PatchCore-style visual anomaly detection system.

---

## 1. Why One-Class Anomaly Detection?

In industrial manufacturing, defective items are rare (often < 0.1% of production), unpredictable, and structurally diverse (e.g. scratches, contaminations, dents, cracks, color bleeding). Standard supervised object detection or segmentation requires thousands of labeled defect examples, which is cost-prohibitive or physically impossible in real-world factories.

One-Class Visual Anomaly Detection models the manifold of **normal, nominal products**. Any sample that deviates significantly from this nominal manifold is flagged as defective.

---

## 2. Why PatchCore-Style?

PatchCore (Roth et al., CVPR 2022) addresses the core limitations of prior anomaly detection paradigms:
- **Autoencoders / GANs**: Often suffer from "over-generalization" (blurring fine textures or accidentally reconstructing unseen defects).
- **Normalizing Flows**: Suffer from heavy computational complexity during density estimation.
- **Deep Feature Memory Banks (PatchCore)**: Directly leverage feature representations from frozen ImageNet networks. They offer state-of-the-art localization resolution without requiring generative training, loss convergence, or hyperparameter instability.

---

## 3. Why ResNet18? Why Layer 2 + Layer 3?

1. **Backbone Choice (ResNet18)**:
   - Provides a balance between inference latency and representational richness.
   - Low memory footprint (44 MB model parameters, fast CPU execution ~140ms per image).
   - Higher capacity backbones (e.g., WideResNet50) offer marginal AUROC gains (+0.5–1.0%) at the cost of 4x memory and 3–5x latency.

2. **Layer Selection (`layer2` + `layer3`)**:
   - `layer1`: Low-level edge and color filters; too localized and noisy for structural anomaly detection.
   - `layer2` (128 channels, $28 \times 28$ grid for $224 \times 224$ input): Preserves fine-grained spatial resolution, capturing scratches and local texture abrasions.
   - `layer3` (256 channels, $14 \times 14$ grid): Captures broader semantic context (e.g., missing caps, misalignments).
   - `layer4`: Highly abstract ImageNet-semantic class features; loses spatial fidelity.
   - **Concatenation**: By bilinearly upsampling `layer3` to $28 \times 28$ and concatenating with `layer2`, we form dense 384-dimensional patch representations that simultaneously capture local texture and global object structure.

---

## 4. What is a Patch Embedding? What is a Memory Bank?

- **Patch Embedding**: A feature vector representing an effective receptive field on the input image. For an input of $224 \times 224$, the model generates $28 \times 28 = 784$ patch embeddings, each of dimension $D = 384$.
- **Memory Bank**: The collection of all patch embeddings extracted from normal training images. For 167 images, the raw memory bank contains $167 \times 784 = 130,928$ vectors.

---

## 5. Why Coreset Selection?

A raw memory bank of 130,928 vectors per category creates prohibitive runtime latency during nearest-neighbor search.

**Greedy K-Center Coreset Selection**:
- Finds a subset $\mathcal{M}_C \subset \mathcal{M}$ of size $K$ that minimizes the maximum distance from any point in $\mathcal{M}$ to its nearest neighbor in $\mathcal{M}_C$:
$$\min_{\mathcal{M}_C} \max_{p \in \mathcal{M}} \min_{c \in \mathcal{M}_C} \|p - c\|_2$$
- **Johnson-Lindenstrauss Random Projection**: Calculating greedy minimax distances in 384 dimensions is computationally intensive. We project features into a 64-dimensional space strictly for index selection:
$$p_{\text{proj}} = \frac{1}{\sqrt{64}} p \cdot W, \quad W \in \mathbb{R}^{384 \times 64}, \quad W_{ij} \sim \mathcal{N}(0, 1)$$
- **Preservation of Original Dimension**: Once the $K$ optimal indices are selected, we extract the corresponding **original 384-dimensional vectors** to construct the final `MemoryBank`.

---

## 6. Why Nearest-Neighbor Scoring & Gaussian Smoothing?

1. **Nearest-Neighbor Anomaly Distance**:
   - For any query patch $p$, its anomaly distance is:
$$d(p) = \min_{m \in \mathcal{M}_C} \|p - m\|_2$$
   - Normal patches lie close to existing patches in the memory bank ($d(p)$ is small). Defective patches (scratches, cracks) deviate from normal representations ($d(p)$ is large).

2. **Gaussian Smoothing**:
   - Raw patch distance maps contain high-frequency spatial noise.
   - Applying a Gaussian filter ($\sigma = 1.0$) propagates anomaly energy across neighboring patches, smoothing out noise and reinforcing true spatial defect clusters.

3. **Image Score as 99th Percentile**:
   - Rather than taking the single maximum pixel ($\max$, which is brittle to outlier noise) or the global average ($\text{mean}$, which dilutes tiny defects like needle holes), the 99th percentile provides a robust compromise: it targets the core defect peak while rejecting isolated noise.

---

## 7. Operational Policy: Normal-only AUTO_PASS

Trong lúc build model chỉ có normal reference, vì vậy V1 chỉ khóa:
- `auto_pass_threshold = Quantile(normal_scores, 0.99)`.
- `pixel_threshold = Quantile(all_normal_heatmap_pixels, 0.99)`.
- score dưới ngưỡng → `AUTO_PASS`; score còn lại → `HUMAN_REVIEW`.

P99 trên cohort calibration nhỏ là heuristic normal-only upper-tail threshold,
không phải cam kết false-reject rate production là 1%. V1 không suy ra
`FAIL_MINOR` hay `FAIL_MAJOR`; mọi ảnh vượt AUTO_PASS đều chuyển HUMAN_REVIEW để
QC quyết định `QC_PASS` hoặc `QC_REJECT`.

---

## 8. Ablation Studies

### Explicit Coreset Size (Category: `bottle`)

| Coreset Size ($K$) | RAM Footprint | Inference Latency (Batch=1) | Image AUROC | Pixel AUROC | AUPRO@0.3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **130,928 (full)** | 191.3 MB | 1,120 ms | 1.0000 | 0.9825 | 0.9422 |
| **4,000** | 5.86 MB | 260 ms | 1.0000 | 0.9821 | 0.9416 |
| **2,000** | 2.93 MB | 185 ms | 1.0000 | 0.9819 | 0.9412 |
| **1,000 (engineering baseline)** | **1.46 MB** | **145 ms** | **1.0000** | **0.9818** | **0.9410** |
| **200** | 0.29 MB | 88 ms | 0.9940 | 0.9760 | 0.9280 |

Các con số trên là historical benchmark cần được tái chạy bằng Dev-only ablation
trước khi chọn release. `coreset_size=1000` là engineering baseline, không phải
claim champion được chọn bằng official Test.

### Layer Ablation (Category: `bottle`)

| Configuration | Feature Dim | Image AUROC | Pixel AUROC | AUPRO@0.3 |
| :--- | :---: | :---: | :---: | :---: |
| `layer2` only | 128 | 0.9860 | 0.9710 | 0.9150 |
| `layer3` only | 256 | 0.9920 | 0.9680 | 0.9020 |
| **`layer2` + `layer3`** | **384** | **1.0000** | **0.9818** | **0.9410** |

**Finding**: Combining `layer2` (high spatial resolution) with `layer3` (semantic context) outperforms either layer individually across both detection (+0.8–1.4%) and localized segmentation (+2.6–3.9% AUPRO).
