Image Catalog Vector Database Generator

Overview

This script is a high-performance ingestion pipeline designed to convert a raw directory of images into a searchable vector database. It is built for deployment-ready environments, focusing on fast inference and decoupled storage.

At a high level, the pipeline performs the following sequence:

    Batch Loading: Scans a target directory for image files and groups them into configurable batches to maximize memory efficiency.

    Preprocessing: Passes the raw images through a Hugging Face AutoProcessor to handle resizing, cropping, and tensor normalization.

    Hardware-Accelerated Inference: Feeds the normalized image tensors into a pre-compiled ONNX vision model to extract dense feature embeddings.

    L2 Normalization: Normalizes the resulting embedding vectors mathematically, setting them up for highly accurate Cosine Similarity searches.

    Dual-Storage Export: * Compiles the vectors into a FAISS Index (IndexFlatIP for Inner Product/Cosine similarity) for sub-millisecond nearest-neighbor math.

        Maps the vector IDs to their original file paths in a lightweight SQLite Database to maintain persistent state and metadata routing.

Architecture Specifications: ONNX and CUDA

When processing large catalogs of visual data (like thousands of product images), native PyTorch inference carries unnecessary overhead designed for training (autograd engines, massive memory footprints). This script transitions the workload to ONNX Runtime (ORT) paired with CUDA Execution.
Why We Used CUDA via ONNX Runtime

    Maximum Throughput: CUDA allows the script to parallelize the massive matrix multiplications required by the vision encoder across the thousands of cores on an NVIDIA GPU. Pairing this with ONNX Runtime strips away Python overhead, executing the computational graph directly via optimized C++ backends.

    Deterministic Resource Allocation: Native PyTorch is notorious for greedy VRAM allocation. ONNX Runtime provides tighter control over memory execution, preventing Out-Of-Memory (OOM) crashes when ingesting large datasets and allowing us to reliably scale the --batch-size.

    Deployment Portability: By using ONNX, the pipeline is entirely decoupled from the original training framework. The same .onnx artifact generating these embeddings can be seamlessly lifted and dropped into a Triton Inference Server later for live production traffic without altering the weights or logic.

How CUDA is Implemented in the Code

The hardware acceleration is explicitly managed through the ort.InferenceSession execution providers.

    Provider Detection: The script checks the host hardware using ort.get_available_providers().

    Fallback Safety: It defaults to the CUDAExecutionProvider if --provider cuda or auto is passed. If a GPU is not detected, or if the user lacks the onnxruntime-gpu library, it gracefully falls back to the CPUExecutionProvider so the script doesn't fatally crash.

    Graph Execution: ```python
    session = ort.InferenceSession(str(vision_onnx), providers=["CUDAExecutionProvider"])

    When `session.run()` is called, the batched `np.float32` pixel arrays are shipped to the GPU's VRAM. The ONNX graph executes the forward pass on the CUDA cores, and the resulting multi-dimensional embedding vectors are pulled back to system RAM as a NumPy array to be stacked and ingested by FAISS.

Usage Example

To run the pipeline on a machine with an NVIDIA GPU, ensure you have onnxruntime-gpu installed, then execute:
Bash

python build_vector_db.py \
    --dataset-dir /path/to/images \
    --vision-onnx /path/to/model/vision_encoder.onnx \
    --processor-dir /path/to/huggingface/processor \
    --out-dir ./VectorDB \
    --batch-size 128 \
    --provider cuda \
    --recursive