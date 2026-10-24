# Image Qwen3-VL-4B (model di dalam image)

Build lewat Runpod (tanpa Docker lokal):
1. Taruh `Dockerfile` ini di root sebuah repo GitHub (boleh private).
2. Runpod console -> Serverless -> New Endpoint -> "Import Git Repository" -> pilih repo itu.
3. GPU 16-24 GB (A4000/A4500/3090/A5000/L4/4090), Scaling: Request Count = 1, Max workers 1, Idle timeout 5 dtk.
4. Environment variable HF_TOKEN tidak diperlukan (model publik, sudah ada di image).

Catatan: ukuran image besar (puluhan GB), build pertama bisa lama.
