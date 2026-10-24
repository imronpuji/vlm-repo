# llama.cpp server (CUDA) + Qwen3-VL-4B GGUF di dalam image + handler Runpod serverless.
# Boot jauh lebih ringan daripada vLLM: tanpa kompilasi/profiling, model dibaca langsung dari disk.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages runpod huggingface_hub requests \
 || pip3 install --no-cache-dir runpod huggingface_hub requests

# Model + mmproj (encoder gambar) diunduh saat build, bukan saat boot
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ggml-org/GLM-OCR-GGUF', allow_patterns=['*Q8_0*'], local_dir='/models')" \
 && ls -la /models

COPY handler.py /handler.py
ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
