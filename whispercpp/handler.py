import base64, subprocess, time, threading, requests, runpod

PORT = 8081
MODEL = "/models/ggml-small-q5_1.bin"
LOG = "/tmp/whisper-server.log"
logf = open(LOG, "w")
proc = subprocess.Popen(["whisper-server", "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--convert", "-t", "4"],
                        stdout=logf, stderr=subprocess.STDOUT)
state = {"ready": False}

def tail(n=25):
    try:
        return "".join(open(LOG).readlines()[-n:])[-2500:]
    except Exception:
        return ""

def waiter():
    t0 = time.time()
    while proc.poll() is None:
        try:
            requests.get(f"http://127.0.0.1:{PORT}/", timeout=2)
            state["ready"] = True
            print(f"whisper-server siap dalam {time.time() - t0:.1f}s", flush=True)
            return
        except Exception:
            time.sleep(0.3)
    print("whisper-server berhenti:\n" + tail(), flush=True)

threading.Thread(target=waiter, daemon=True).start()

def handler(job):
    # Handler didaftarkan segera; kegagalan server dilaporkan lewat output supaya terlihat dari API.
    t0 = time.time()
    while not state["ready"]:
        if proc.poll() is not None:
            return {"error": f"whisper-server berhenti (kode {proc.returncode})", "log": tail()}
        if time.time() - t0 > 150:
            return {"error": "whisper-server tidak siap dalam 150 dtk", "log": tail()}
        time.sleep(0.2)
    i = job["input"]
    b64 = i.get("audio_base64") or i.get("audio")
    if not b64:
        return {"error": "audio_base64 wajib"}
    form = {"response_format": "verbose_json", "temperature": "0.0"}
    if i.get("language"):
        form["language"] = i["language"]
    if i.get("initial_prompt"):
        form["prompt"] = i["initial_prompt"]
    r = requests.post(f"http://127.0.0.1:{PORT}/inference", files={"file": ("audio", base64.b64decode(b64))}, data=form, timeout=600)
    if not r.ok:
        return {"error": f"whisper-server HTTP {r.status_code}: {r.text[:200]}", "log": tail(10)}
    d = r.json()
    segs = [{"id": s.get("id", n), "start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
            for n, s in enumerate(d.get("segments", []))]
    text = (d.get("text") or "".join(s["text"] for s in segs)).strip()
    return {"transcription": text, "segments": segs, "detected_language": d.get("language")}

runpod.serverless.start({"handler": handler})
