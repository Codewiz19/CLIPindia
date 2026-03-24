# ClipIndia Execution Blueprint (Windows 11, RTX 3050 4GB)

## 1) System-Aware Goal
Build and deploy a **visual fashion search engine** for Indian products using a resource-aware setup for:
- Windows 11 laptop
- 16 GB RAM
- RTX 3050 (4 GB VRAM)

Because VRAM is limited, this plan uses:
- mixed precision (FP16)
- small batch sizes + gradient accumulation
- staged training (freeze -> partial unfreeze)
- optional lower-resolution warmup (192) before 224

---

## 2) Project Milestones (What “done” means)
1. Dataset prepared and cleaned.
2. Baseline retrieval metrics recorded (`Recall@1/5/10/20`).
3. Fine-tuned model improves `Recall@10` significantly.
4. FAISS index built for full catalog.
5. FastAPI + Gradio demo works locally.
6. Multi-modal search (Image + Text) works with alpha slider.
7. Docker-ready and optionally deployable.

---

## 3) Recommended Folder Structure

```text
ClipIndia/
  Blue Print/
    Execution-Plan.md
  data/
    raw/
    cleaned/
  notebooks/
  src/
  configs/
  checkpoints/
  index/
  results/
  logs/
  requirements.txt
  README.md
```

---

## 4) Environment Setup (Windows 11)

### Step 4.1: Create/activate venv
- Use Python 3.10 or 3.11 (preferred for library compatibility).
- Create venv in project root.

### Step 4.2: Install core packages
- `torch`, `torchvision` (CUDA build)
- `transformers`, `datasets`, `accelerate`
- `faiss-cpu` (start CPU index first for stability)
- `fastapi`, `uvicorn`, `gradio`
- `pandas`, `numpy`, `scikit-learn`, `Pillow`, `matplotlib`, `tqdm`

### Step 4.3: Verify GPU
- Confirm CUDA is available in PyTorch.
- Log GPU name and memory at startup.

---

## 5) Data Pipeline

### Step 5.1: Download dataset
- Myntra Fashion Product Images (Small) from Kaggle.
- Place images in `data/raw/images` and metadata CSV in `data/raw`.

### Step 5.2: Clean data
- Remove rows with missing image file.
- Remove rows with null `articleType`.
- Remove categories with extremely low support (e.g., <5 samples).
- Verify image readability (PIL open test).

### Step 5.3: Build text prompts
Template example:
- `"a photo of a {baseColour} {articleType} for {gender}, {usage} wear"`

### Step 5.4: Split strategy
- Stratified split by `articleType`:
  - Train 80%
  - Validation 10%
  - Test 10%
- Save all split CSVs to `data/cleaned/`.

---

## 6) Baseline (Before Fine-Tuning)

### Step 6.1: Load pretrained SigLIP/SigLIP-2 variant
For 4GB VRAM:
- Prefer smaller variant first (or low batch + FP16).

### Step 6.2: Evaluate retrieval baseline
- Build train embeddings.
- Query test images.
- Compute `Recall@1/5/10/20`.
- Save to `results/baseline_results.json`.

This is your reference point. Do not change test split after this.

---

## 7) Fine-Tuning Plan for 4GB VRAM

### Step 7.1: Training settings (safe defaults)
- Resolution: start `192x192`, then `224x224` in later run.
- Batch size: `4` (or `2` if OOM).
- Gradient accumulation: `8` (effective batch 32).
- Mixed precision: `fp16=True`.
- Epochs: 4–8 initially.
- Optimizer: AdamW.
- LR: `1e-5` to `3e-5` search.
- Weight decay: `0.01`.
- Gradient clipping: `1.0`.

### Step 7.2: Freeze strategy
1. Epoch 1–2: freeze most vision layers, train projection/head layers.
2. Epoch 3+: unfreeze top transformer blocks only.

This avoids OOM and unstable training.

### Step 7.3: OOM fallback ladder
If out-of-memory occurs:
1. Reduce batch size `4 -> 2`.
2. Increase accumulation steps.
3. Reduce image size `224 -> 192`.
4. Enable gradient checkpointing.
5. Keep more layers frozen.

### Step 7.4: Save checkpoints
- Save best model by validation loss.
- Save training config + metrics for reproducibility.

---

## 8) Evaluation and Experiment Tracking

### Step 8.1: Run same test protocol
- Same test set.
- Same metric computation.
- Compare directly vs baseline.

### Step 8.2: Track runs
Store each run:
- hyperparameters
- best val loss
- Recall@K
- latency notes

Files:
- `results/finetuned_run01_results.json`
- `results/finetuned_run02_results.json`

---

## 9) Build Search Index (FAISS)

### Step 9.1: Encode full product catalog
- Use best fine-tuned checkpoint.
- Batch inference without gradients.
- Normalize embeddings to unit vectors.

### Step 9.2: Build index
- Start with `IndexFlatIP`.
- Save index to `index/product.index`.
- Save ID-metadata map to `index/product_metadata.pkl`.

Note: For 44K vectors, CPU FAISS is acceptable on this laptop.

---

## 10) API + Demo (Local First)

### Step 10.1: FastAPI backend
Endpoints:
- `GET /health`
- `POST /search` (image upload, top-k retrieval)

### Step 10.2: Gradio frontend
- Image upload
- Top-k result cards (image, name, score)

### Step 10.3: Performance target (realistic on this machine)
- Initial local target: ~150–300ms/query
- After optimization: ~80–150ms/query possible (depends on model size and image preprocessing)

---

## 11) Optimization Steps (After MVP)
1. Enable Torch compile/inference optimizations where stable.
2. Export vision path to ONNX and benchmark.
3. Preload model/index once at startup.
4. Reduce preprocessing overhead and disk reads.
5. Add simple in-memory cache for repeated queries.

---

## 12) Weekly Action Plan

### Week 1
- Environment setup
- Data cleaning
- EDA notebook
- Baseline metrics lock

### Week 2
- Fine-tuning run 1
- Evaluate and compare
- Fine-tuning run 2 (if needed)

### Week 3
- Build FAISS full index
- FastAPI + Gradio integration
- Local demo complete

### Week 4
- Latency optimization + ONNX trial
- Documentation + results plots
- Final README + demo recording

### Week 5
- Add multi-modal search module (`src/multimodal_search.py`)
- Add `POST /search/multimodal` endpoint
- Update Gradio UI: text modifier + alpha slider + mode toggle

### Week 6
- Multi-modal qualitative evaluation + alpha ablation
- Latency comparison (image-only vs image+text)
- Final demo polish and deployment update

---

## 13) Risks and Mitigation (Your Hardware)

### Risk A: GPU OOM (most likely)
Mitigation: lower batch, accumulation, smaller image size, partial unfreeze.

### Risk B: Slow training on laptop
Mitigation: shorter pilot runs first, then final full run.

### Risk C: Thermal throttling
Mitigation: train in cooler hours, keep charger connected, use performance mode and adequate cooling.

### Risk D: Windows path issues
Mitigation: use `pathlib`, avoid hardcoded separators, keep relative paths in config.

---

## 14) Deliverables Checklist
- [ ] Cleaned dataset and split CSVs
- [ ] Baseline JSON results
- [ ] Fine-tuned checkpoint + config
- [ ] Recall@K comparison table and chart
- [ ] FAISS index + metadata map
- [ ] FastAPI backend
- [ ] Gradio demo
- [ ] Multi-modal search endpoint + UI integration
- [ ] Alpha ablation chart and multimodal examples
- [ ] Final README with architecture and results

---

## 15) Practical Hyperparameter Starter Pack (for your laptop)
- `image_size=192`
- `train_batch_size=4`
- `eval_batch_size=8`
- `gradient_accumulation_steps=8`
- `learning_rate=2e-5`
- `weight_decay=0.01`
- `num_epochs=5`
- `fp16=True`
- `max_grad_norm=1.0`
- `num_workers=2` (Windows-safe start)

If stable, move to `image_size=224` and tune for best Recall@10.

---

## 16) Multi-Modal Integration Plan (Image + Text)

### 16.1 Why this is a good idea
Yes, this is a very good idea.
- It makes your demo noticeably more advanced than pure similarity search.
- It is practical: **no retraining required** if your SigLIP checkpoint is already ready.
- It improves product usefulness (e.g., “this kurta but in red”).

### 16.2 Integration sequence
1. Create `src/multimodal_search.py` for image embedding + text embedding + weighted combine.
2. Add `POST /search/multimodal` in `src/api.py`.
3. Update `src/demo.py` with:
  - text modifier input
  - alpha slider (`0.0` to `1.0`)
  - mode toggle (`Image Only`, `Image + Text`, `Text Only`)
4. Keep existing `/search` endpoint unchanged for backward compatibility.

### 16.3 Query equation
Use normalized embeddings and:

`query = normalize(alpha * image_emb + (1 - alpha) * text_emb)`

Recommended defaults:
- `alpha=0.7` (image dominant)
- fallback to pure image when text is empty

### 16.4 Validation plan for multi-modal
- Save 10 side-by-side examples: image-only vs image+text.
- Run alpha sweep: `0.3, 0.5, 0.7, 0.9`.
- Log which alpha gives best qualitative consistency.

---

## 17) Kaggle GPU Training Strategy (Recommended)

### 17.1 Is Kaggle a good idea?
Yes — for your setup, Kaggle is a good idea for training.

Why:
- Your local RTX 3050 4GB is tight for stable fine-tuning.
- Kaggle GPUs are faster and reduce thermal/power issues on laptop.
- You can still keep inference/API development local on Windows.

### 17.2 Suggested split of work
- **Kaggle:** heavy jobs (training, long evaluations, checkpoint generation)
- **Local Windows:** API, FAISS serving, Gradio, integration testing, demo

### 17.3 Kaggle workflow (step-by-step)
1. Create Kaggle notebook with GPU enabled.
2. Upload or link dataset (Kaggle dataset or competition dataset).
3. Install project dependencies in notebook.
4. Run training with fixed seeds and config saved to JSON/YAML.
5. Save outputs:
  - best checkpoint
  - training logs
  - eval results JSON
6. Export artifacts to Kaggle Output.
7. Download artifacts and place locally:
  - `checkpoints/`
  - `results/`
8. Rebuild FAISS index locally from best checkpoint (or on Kaggle if faster, then download index).

### 17.4 Kaggle constraints to design around
- Session time limits: save checkpoints every epoch.
- Ephemeral runtime: never keep only in `/kaggle/working` without exporting outputs.
- Internet/package limits can vary: pin versions in `requirements.txt`.

### 17.5 Reproducibility rules
- Keep one master config per run in `configs/`.
- Name runs clearly: `run_01`, `run_02`, `run_03`.
- Store:
  - git commit hash (if available)
  - dataset version
  - exact hyperparameters
  - final metrics

### 17.6 Practical recommendation
Train on Kaggle first, then integrate and deploy from local machine. This is the best cost/performance path for your current hardware.

---

This blueprint is optimized for your current system so execution is practical, stable, and interview-ready.