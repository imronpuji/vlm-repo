# llama.cpp server (CUDA) + Spark-X2.6-4B GGUF (Q8_0) di dalam image + handler Runpod serverless (streaming).
# Model butuh llama.cpp >= b10828 (arsitektur spark2_5); image server-cuda terbaru memenuhi.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages runpod huggingface_hub requests \
 || pip3 install --no-cache-dir runpod huggingface_hub requests

# Model diunduh saat build, bukan saat boot
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('XHToken/Spark-X2.6-4B-GGUF', allow_patterns=['*Q8_0*'], local_dir='/models')" \
 && ls -la /models

COPY handler.py /handler.py
ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
