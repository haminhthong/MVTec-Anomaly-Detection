# Kiến trúc và phương pháp PatchCore-style

Tài liệu này giải thích cơ sở kỹ thuật và operational policy của hệ thống PatchCore-style. README là contract vận hành; tài liệu này chỉ đi sâu vào model.

---

## 1. Why One-Class Anomaly Detection?

MVTec AD cung cấp `train/good` normal và official test có defect/mask. Vì vậy normal-only anomaly detection là formulation phù hợp cho repo này. Supervised learning vẫn có thể phù hợp trong nhà máy nếu có đủ defect labels ổn định; repo hiện tại không dùng supervised classification.

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
   - Cân bằng giữa chất lượng biểu diễn và chi phí suy luận; latency thực tế phải đo bằng `scripts/benchmark_inference.py` vì phụ thuộc hardware/runtime.
   - Backbone lớn hơn có thể cho biểu diễn khác nhưng phải được đánh giá lại trên Dev/Calibration; không suy ra mức tăng cố định từ benchmark `bottle`.

2. **Layer Selection (`layer2` + `layer3`)**:
   - `layer1`: Low-level edge and color filters; too localized and noisy for structural anomaly detection.
   - `layer2` (128 channels, $28 \times 28$ grid for $224 \times 224$ input): Preserves fine-grained spatial resolution, capturing scratches and local texture abrasions.
   - `layer3` (256 channels, $14 \times 14$ grid): Captures broader semantic context (e.g., missing caps, misalignments).
   - `layer4`: Highly abstract ImageNet-semantic class features; loses spatial fidelity.
   - **Concatenation**: By bilinearly upsampling `layer3` to $28 \times 28$ and concatenating with `layer2`, we form dense 384-dimensional patch representations that simultaneously capture local texture and global object structure.

---

## 4. What is a Patch Embedding? What is a Memory Bank?

- **Patch Embedding**: A feature vector representing an effective receptive field on the input image. For an input of $224 \times 224$, the model generates $28 \times 28 = 784$ patch embeddings, each of dimension $D = 384$.
- **Memory Bank**: Tập patch embeddings lấy từ ảnh normal Reference. Với $N$ ảnh và grid $H \times W$, full memory có $N \times H \times W$ vector trước khi coreset; con số cụ thể phụ thuộc dataset và preprocessing.

---

## 5. Why Coreset Selection?

Full memory có thể lớn theo số ảnh và kích thước feature grid, vì vậy repo dùng coreset để kiểm soát chi phí nearest-neighbor search.

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

## 8. Ablation trên Dev

Ablation phải chạy trên Reference/Dev và synthetic stress; official test chỉ dùng ở
locked evaluation. Các kết quả sinh ra trong `experiments/` là evidence để chọn
configuration, không tự động cập nhật production pointer.

```bash
python scripts/run_ablations.py --category bottle --experiment coreset
python scripts/run_ablations.py --category bottle --experiment layers
python scripts/run_ablations.py --category bottle --experiment backbone
```

Các trục có trong code:

- `coreset`: so sánh các giá trị K cụ thể.
- `layers`: so sánh `layer2`, `layer3` và `layer2 + layer3`.
- `backbone`: so sánh các backbone đã đăng ký.

Sau khi chọn cấu hình, phải build release mới với `model_version` mới, chạy locked
evaluation và review report trước khi cập nhật production.
