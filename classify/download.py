from transformers import pipeline
pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
