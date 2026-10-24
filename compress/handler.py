import time, torch, runpod
from llmlingua import PromptCompressor

dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
comp = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                        use_llmlingua2=True, device_map=dev)
print(f"model siap {time.time()-t0:.1f}s device={dev}", flush=True)

def handler(job):
    i = job["input"]
    text = i.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "text wajib"}
    if len(text) > 300_000:
        return {"error": "text terlalu panjang (maks 300.000 karakter)"}
    rate = min(0.9, max(0.1, float(i.get("rate", 0.5))))
    r = comp.compress_prompt(text, rate=rate, force_tokens=["\n", ".", "?", "!"], drop_consecutive=True)
    return {"compressed": r["compressed_prompt"], "origin_tokens": r["origin_tokens"],
            "compressed_tokens": r["compressed_tokens"]}

runpod.serverless.start({"handler": handler})
