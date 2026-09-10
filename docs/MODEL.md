# Model và scoring

## PatchCore-style components

- ResNet18 pretrained nhưng frozen, không backpropagation và không fine-tune.
- `layer2` + `layer3` tạo embedding đa tỷ lệ.
- Greedy k-center giảm toàn bộ normal patches thành coreset.
- Memory bank lưu patch feature gốc để so sánh runtime.
- 1-NN Euclidean distance tạo anomaly map.
- Gaussian smoothing làm map ổn định hơn trước khi tính image score.

## Artifact

```text
models/bottle/
├── memory_bank.npy
└── metadata.json
```

`metadata.json` lưu backbone, weights, feature layers, preprocessing, coreset
shape, scoring, calibration, thresholds và dataset fingerprint. Split file và
reference file nằm ở `reports/`, không được detector đọc khi serving.

## Threshold

```text
image_threshold = quantile(normal_calibration_scores, image_quantile)
pixel_threshold = quantile(all_normal_calibration_pixels, pixel_quantile)
```

Đây là heuristic trên normal holdout, không phải guarantee false-reject rate
trên dữ liệu mới. Model chỉ tạo triage decision; nó không suy ra defect type,
severity hay QC outcome.

## Khác biệt với PatchCore paper

Đây là implementation lấy cảm hứng từ PatchCore cho mục tiêu học tập và
benchmark MVTec. Projection 64 chiều chỉ dùng trong bước chọn coreset; runtime
vẫn lưu feature ResNet18 ghép từ layer2/layer3 với dimension 384. Vì vậy README
không gọi đây là reproduction nguyên bản 100%.
