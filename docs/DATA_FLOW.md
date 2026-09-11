# Data flow

## 1. Dataset boundary

```text
data/raw/<category>/train/good
        |
        +--> Reference: xây memory bank
        +--> Dev: ablation và synthetic stress
        +--> Calibration: image/pixel threshold

data/raw/<category>/test + ground_truth
        |
        +--> chỉ evaluate_category() đọc ở bước cuối
```

`validate_reference_category()` không đọc test hoặc ground truth. `validate_evaluation()`
kiểm tra test/mask sau khi model đã được tạo.

## 2. Feature và score

1. Ảnh được resize/normalize theo `PreprocessingConfig`.
2. ResNet18 frozen trả feature map `layer2` và `layer3`.
3. Feature map được căn chỉnh cùng kích thước rồi flatten thành patch embeddings.
4. Greedy k-center chọn tối đa 1,000 patch đại diện.
5. Mỗi patch ảnh mới tìm khoảng cách Euclidean tới memory bank bằng 1-NN.
6. Khoảng cách được reshape theo lưới patch và Gaussian smoothing.
7. Percentile của heatmap là image anomaly score.

## 3. Runtime decision

```text
image
  -> input check
  -> RECAPTURE_REQUIRED nếu không đạt
  -> feature extraction
  -> nearest-neighbor distances
  -> heatmap + image score
  -> score < image_threshold: PASS_CANDIDATE
  -> ngược lại: REVIEW_REQUIRED
  -> human QC: QC_PASS hoặc QC_REJECT
```

`image_threshold` và `pixel_threshold` được lưu trong `metadata.json`. Quantile
P99 chỉ là heuristic upper-tail từ normal Calibration. API kiểm tra `/live` cho
process-level liveness và `/ready` cho model readiness; detector theo category
được cache sau lần khởi tạo đầu tiên.
