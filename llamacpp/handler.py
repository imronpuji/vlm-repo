import glob, os, shutil, subprocess, sys, time
import requests
import runpod

PORT = 8080
BASE = f"http://127.0.0.1:{PORT}"


def pick(patterns, exclude=None):
    for pat in patterns:
        for f in sorted(glob.glob(pat, recursive=True)):
            if exclude and exclude in os.path.basename(f).lower():
                continue
            return f
    raise SystemExit(f"file GGUF tidak ditemukan: {patterns}")


model = pick(["/models/**/*Q4_K_M*.gguf"], exclude="mmproj")
mmproj = pick(["/models/**/*mmproj*Q8_0*.gguf", "/models/**/*mmproj*.gguf"])
binary = shutil.which("llama-server") or "/app/llama-server"
cmd = [binary, "-m", model, "--mmproj", mmproj, "--host", "127.0.0.1", "--port", str(PORT),
       "-ngl", "99", "-c", "8192", "-np", "2"]
print("start:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd)

t0 = time.time()
while True:  # worker baru dianggap siap setelah llama-server selesai memuat model
    if proc.poll() is not None:
        sys.exit("llama-server berhenti saat start")
    try:
        if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
            break
    except Exception:
        pass
    time.sleep(0.3)
print(f"llama-server siap dalam {time.time() - t0:.1f}s", flush=True)


def handler(job):
    inp = job.get("input", {}) or {}
    route = inp.get("openai_route", "/v1/chat/completions")
    body = dict(inp.get("openai_input", inp))
    body["stream"] = False  # gateway memakai respons non-stream untuk model gambar
    r = requests.post(BASE + route, json=body, timeout=600)
    try:
        out = r.json()
    except Exception:
        return {"error": r.text[:500]}
    if r.status_code != 200:
        return {"error": out}
    return out


runpod.serverless.start({"handler": handler})
