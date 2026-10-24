import base64, io, os, tempfile, time, torch, torchaudio as ta, runpod
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
print(f"model siap {time.time()-t0:.1f}s device={dev}", flush=True)
LANGS = set(model.get_supported_languages().keys())

def handler(job):
    i = job["input"]
    text = (i.get("text") or "").strip()
    if not text or len(text) > 800:
        return {"error": "text wajib, maks 800 karakter"}
    lang = (i.get("language") or "en").lower()
    if lang not in LANGS:
        return {"error": f"language tidak didukung: {lang}", "supported": sorted(LANGS)}
    ref = None
    if i.get("ref_audio_base64"):
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(base64.b64decode(i["ref_audio_base64"])); f.close(); ref = f.name
    try:
        t1 = time.time()
        wav = model.generate(text, language_id=lang, audio_prompt_path=ref,
                             cfg_weight=float(i.get("cfg_weight", 0.3)), exaggeration=float(i.get("exaggeration", 0.5)))
        buf = io.BytesIO(); ta.save(buf, wav.cpu(), model.sr, format="wav")
        return {"audio_base64": base64.b64encode(buf.getvalue()).decode(), "sample_rate": model.sr,
                "seconds": round(wav.shape[-1] / model.sr, 2), "gen_seconds": round(time.time() - t1, 2)}
    finally:
        if ref: os.unlink(ref)

runpod.serverless.start({"handler": handler})
