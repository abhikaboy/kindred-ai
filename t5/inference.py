# Trained Model and Tokenizer
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np

model = T5ForConditionalGeneration.from_pretrained("../results")
tokenizer = T5Tokenizer.from_pretrained("../results")

user_input = "Check emails at 8:00 AM every weekday."

def sentence_to_task(sentence, model, tokenizer):
    input_text = "covert1 to json: " + sentence
    # tokenize input
    input_ids = tokenizer(input_textm, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    # decode 
    json_string = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return json_string

task = input("Give a Task:")
print(sentence_to_task(task, model, tokenizer))
