# Worker NLP multibahasa: sentimen + klasifikasi zero-shot. Model sudah ada di dalam image.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN pip install --no-cache-dir runpod "transformers==4.46.3" sentencepiece protobuf
ENV HF_HOME=/models TRANSFORMERS_OFFLINE=0
COPY download.py /download.py
RUN python /download.py
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
