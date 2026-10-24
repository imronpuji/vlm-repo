# Worker vLLM Runpod dengan model Qwen3-VL-4B sudah ada DI DALAM image,
# jadi cold start tidak mengunduh ~9 GB dari Hugging Face lagi.
FROM runpod/worker-v1-vllm:v2.27.0

# Model disimpan di dalam image
ENV HF_HOME=/models
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct')" \
 && du -sh /models

# Default worker: baca dari cache lokal (tanpa jaringan), tanpa CUDA graph supaya boot lebih cepat
ENV HF_HUB_OFFLINE=1 \
    MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct \
    OPENAI_SERVED_MODEL_NAME_OVERRIDE=qwen3-vl-4b \
    MAX_MODEL_LEN=8192 \
    MAX_NUM_SEQS=16 \
    MAX_CONCURRENCY=16 \
    GPU_MEMORY_UTILIZATION=0.92 \
    ENFORCE_EAGER=true \
    TRUST_REMOTE_CODE=1
