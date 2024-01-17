from transformers import pipeline


classifier = pipeline(
    "text-classification",
    model="facebook/roberta-hate-speech-dynabench-r4-target",
    tokenizer="facebook/roberta-hate-speech-dynabench-r4-target",
)
classifier.save_pretrained("./models/")
