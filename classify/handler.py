import time, torch, runpod
from transformers import pipeline

dev = 0 if torch.cuda.is_available() else -1
dt = torch.float16 if dev == 0 else torch.float32
t0 = time.time()
sent = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=dev, torch_dtype=dt, top_k=None, truncation=True, max_length=256)
zs = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
              device=dev, torch_dtype=dt)
print(f"model siap {time.time()-t0:.1f}s device={dev}", flush=True)

MAX_TEXTS, MAX_CHARS = 100, 2000

def handler(job):
    i = job["input"]
    texts = i.get("texts") or ([i["text"]] if i.get("text") else [])
    if not texts or len(texts) > MAX_TEXTS:
        return {"error": f"texts harus 1..{MAX_TEXTS}"}
    texts = [str(t)[:MAX_CHARS] for t in texts]
    task = i.get("task", "sentiment")
    if task == "sentiment":
        out = sent(texts, batch_size=16)
        res = []
        for scores in out:
            scores = sorted(scores, key=lambda s: -s["score"])
            res.append({"label": scores[0]["label"].lower(),
                        "scores": {s["label"].lower(): round(s["score"], 4) for s in scores}})
        return {"results": res}
    if task == "classify":
        labels = i.get("labels")
        if not labels or not (2 <= len(labels) <= 30):
            return {"error": "labels harus 2..30"}
        out = zs(texts, candidate_labels=[str(l) for l in labels],
                 multi_label=bool(i.get("multi_label", False)), batch_size=8)
        if isinstance(out, dict):
            out = [out]
        return {"results": [{"label": o["labels"][0],
                             "scores": {l: round(s, 4) for l, s in zip(o["labels"], o["scores"])}} for o in out]}
    return {"error": "task tidak dikenal"}

runpod.serverless.start({"handler": handler})
