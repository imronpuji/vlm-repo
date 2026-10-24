# Kompresi prompt LLMLingua-2 (multibahasa) dengan model di dalam image.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN pip install --no-cache-dir runpod llmlingua==0.2.2 "transformers==4.46.3" sentencepiece
ENV HF_HOME=/models
COPY download.py /download.py
RUN python /download.py
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
