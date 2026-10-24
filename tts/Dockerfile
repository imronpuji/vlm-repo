# TTS + voice clone (Chatterbox Multilingual, MIT) dengan model di dalam image.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir runpod chatterbox-tts
ENV HF_HOME=/models
COPY download.py /download.py
RUN python /download.py
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
