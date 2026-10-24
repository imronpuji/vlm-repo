from chatterbox.mtl_tts import ChatterboxMultilingualTTS
m = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
print("langs:", sorted(m.get_supported_languages().keys()))
