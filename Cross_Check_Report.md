# Cross-Check Report: ONNX Conversion and Fine-Tuning Validation

Date: 2026-03-24  
Workspace: [ClipIndia](.)

## 1) Objective
Validate two things:
- ONNX export fidelity against corresponding PyTorch fine-tuned models.
- Whether fine-tuning produced meaningful encoder drift versus baseline SigLIP.

## 2) Data and Artifacts Used
- Image evaluation set: [SimilarityData](SimilarityData)
- Fine-tuned HF model: [clipindia_export_bundle/hf_model](clipindia_export_bundle/hf_model)
- Fine-tuned HF processor: [clipindia_export_bundle/hf_processor](clipindia_export_bundle/hf_processor)
- ONNX vision model: [Ai Engine/onnx Model/vision.onnx](Ai%20Engine/onnx%20Model/vision.onnx)
- ONNX text model: [Ai Engine/onnx Model/text.onnx](Ai%20Engine/onnx%20Model/text.onnx)
- Baseline model reference: `google/siglip-base-patch16-224`

## 3) Scripts Used for Validation
- Vision ONNX vs PT parity: [similarity_analysis.py](similarity_analysis.py)
- Baseline vs fine-tuned vision drift: [baseline_vs_finetuned_similarity.py](baseline_vs_finetuned_similarity.py)
- Text ONNX vs PT parity and baseline vs fine-tuned text drift: [text_encoder_comparison.py](text_encoder_comparison.py)

## 4) Results

### A. Vision Encoder: ONNX vs Fine-Tuned PyTorch
- image_count: 80
- embedding_dim: 768
- row_cos_min: 0.9999998807907104
- row_cos_mean: 1.0
- row_cos_max: 1.0000001192092896
- emb_abs_diff_mean: 3.077455446032218e-08
- emb_abs_diff_max: 4.76837158203125e-07
- pairwise_pearson: 0.9999999999907185
- pairwise_mae: 1.1558773849174031e-07
- pairwise_max_abs_diff: 5.960464477539062e-07
- top5_overlap: 1.0
- top10_overlap: 1.0

Interpretation: ONNX vision export is numerically equivalent to fine-tuned PyTorch output (differences are only floating-point noise).

### B. Vision Encoder: Baseline vs Fine-Tuned
- image_count: 80
- embedding_dim: 768
- baseline_to_finetuned_cos_min: 0.7168068885803223
- baseline_to_finetuned_cos_mean: 0.8187229037284851
- baseline_to_finetuned_cos_max: 0.9043655395507812
- emb_abs_diff_mean: 0.015654271468520164
- emb_abs_diff_max: 0.18697190284729004
- pairwise_pearson: 0.751116212110956
- pairwise_mae: 0.23761990666389465
- pairwise_max_abs_diff: 0.43785032629966736
- top5_overlap: 0.61
- top10_overlap: 0.6637500000000001

Interpretation: Fine-tuning produced clear, meaningful drift in vision embedding space.

### C. Text Encoder: ONNX vs Fine-Tuned PyTorch
- item_count: 30
- embedding_dim: 768
- row_cos_min: 0.9999998807907104
- row_cos_mean: 1.0
- row_cos_max: 1.0000001192092896
- emb_abs_diff_mean: 3.745561727441782e-08
- emb_abs_diff_max: 5.066394805908203e-07
- pairwise_pearson: 0.9999999999963324
- pairwise_mae: 1.4332519526760734e-07
- pairwise_max_abs_diff: 5.364418029785156e-07
- top5_overlap: 1.0
- top10_overlap: 1.0

Interpretation: ONNX text export is numerically equivalent to fine-tuned PyTorch output.

### D. Text Encoder: Baseline vs Fine-Tuned
- item_count: 30
- embedding_dim: 768
- row_cos_min: 0.5284855365753174
- row_cos_mean: 0.7256885170936584
- row_cos_max: 0.8645009994506836
- emb_abs_diff_mean: 0.02075188048183918
- emb_abs_diff_max: 0.2900572717189789
- pairwise_pearson: 0.5043611794401553
- pairwise_mae: 0.07591582834720612
- pairwise_max_abs_diff: 0.23367488384246826
- top5_overlap: 0.52
- top10_overlap: 0.6133333333333334

Interpretation: Fine-tuning produced substantial drift in text embedding space.

## 5) Final Conclusion
- ONNX conversion is correct for both encoders (vision and text).
- Fine-tuning was applied effectively and changed both encoders.
- Current remaining question is performance quality against task-level retrieval metrics (e.g., Recall@K/MRR/nDCG in production-style evaluation), not conversion correctness.

## 6) Recommended Next Validation
- Run strict instance-level text→image benchmark comparing baseline vs fine-tuned.
- Report Recall@1/5/10, MRR, and nDCG on real query distributions.
- Keep this report as conversion/training-integrity evidence.
