# Using PyTorch we will build a model to tokenize a sentence and produce 
# a JSON object using the schema specified in "schema.schema.json"

import json
from transformers import AutoTokenizer, DistilBertForTokenClassification, Pipeline
import torch

# Get the model with the pretrained dataset 
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")

# Uses the appropriate tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    return outputs


# Get the data from the JSON file
data = json.loads(open("data.json").read())

# Create a list of sentences
sentences = []
for task in data:
    sentences.append(task["input"])

# Create trainer 
trainer = trainer.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

