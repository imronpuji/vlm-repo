import base64, subprocess, sys, time, requests, runpod

PORT = 8081
MODEL = "/models/ggml-small-q5_1.bin"
proc = subprocess.Popen(["whisper-server", "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--convert", "-t", "4"])
t0 = time.time()
while True:
    if proc.poll() is not None:
        sys.exit("whisper-server berhenti saat start")
    try:
        requests.get(f"http://127.0.0.1:{PORT}/", timeout=2)
        break
    except Exception:
        time.sleep(0.2)
print(f"whisper-server siap dalam {time.time() - t0:.1f}s", flush=True)

def handler(job):
    i = job["input"]
    b64 = i.get("audio_base64") or i.get("audio")
    if not b64:
        return {"error": "audio_base64 wajib"}
    data = base64.b64decode(b64)
    form = {"response_format": "verbose_json", "temperature": "0.0"}
    if i.get("language"):
        form["language"] = i["language"]
    if i.get("initial_prompt"):
        form["prompt"] = i["initial_prompt"]
    r = requests.post(f"http://127.0.0.1:{PORT}/inference", files={"file": ("audio", data)}, data=form, timeout=600)
    if not r.ok:
        return {"error": f"whisper-server HTTP {r.status_code}: {r.text[:200]}"}
    d = r.json()
    segs = [{"id": s.get("id", n), "start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
            for n, s in enumerate(d.get("segments", []))]
    text = (d.get("text") or "".join(s["text"] for s in segs)).strip()
    return {"transcription": text, "segments": segs, "detected_language": d.get("language")}

runpod.serverless.start({"handler": handler})
