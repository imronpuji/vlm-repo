# whisper.cpp (CUDA) + model ggml di dalam image + handler Runpod serverless.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp /src
WORKDIR /src
# A4000/A4500/A5000 = sm_86, RTX 4090/L4 = sm_89; statis supaya tidak perlu menyalin library
RUN cmake -B build -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="86;89" -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)" --target whisper-server

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir runpod requests
COPY --from=build /src/build/bin/whisper-server /usr/local/bin/whisper-server
RUN mkdir -p /models && curl -fL -o /models/ggml-small-q5_1.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin && ls -la /models
COPY handler.py /handler.py
CMD ["python3", "-u", "/handler.py"]
