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


model = pick(["/models/**/*Q8_0*.gguf"], exclude="mmproj")
binary = shutil.which("llama-server") or "/app/llama-server"
cmd = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(PORT),
       "-ngl", "99", "-c", "32768", "-np", "1", "--jinja"]
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
    if body.get("stream"):
        body["stream_options"] = {"include_usage": True}
        r = requests.post(BASE + route, json=body, timeout=600, stream=True)
        if r.status_code != 200:
            yield {"error": r.text[:500]}
            return
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                yield line + "\n\n"
        return
    r = requests.post(BASE + route, json=body, timeout=600)
    try:
        out = r.json()
    except Exception:
        yield {"error": r.text[:500]}
        return
    yield {"error": out} if r.status_code != 200 else out


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
