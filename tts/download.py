import torch
# chatterbox 0.1.3 memuat s3gen.pt tanpa map_location; di mesin build tanpa GPU ini gagal. Paksa ke CPU hanya saat build.
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": k.get("map_location", "cpu")})
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
m = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
print("langs:", sorted(m.get_supported_languages().keys()))
