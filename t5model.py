from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load the JSON dataset 
dataset = json.loads(open("data.json").read())

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

inputs = [entry["input"] for entry in dataset]
outputs = [json.dumps(entry["output"]) for entry in dataset]

hf_dataset = Dataset.from_dict({
  "input": inputs, 
  "output": outputs
})


train_test = hf_dataset.train_test_split(test_size=0.2)

def preprocess(entries):
    model_inputs = tokenizer(
        # Important to tell the model specifically what transformation to preform
        ["convert to json: " + ex for ex in entries["input"]],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    # Tokenize outputs
    labels = tokenizer(
        entries["output"],
        max_length=512,
        padding="max_length",
        truncation=True
    ).input_ids
    
    model_inputs["labels"] = labels
    return model_inputs

# Apply preprocessing
tokenized_datasets = train_test.map(
    preprocess,
    batched=True,
    remove_columns=["input", "output"]
)

# Model 
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    fp16=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train model
trainer.train()



def sentence_to_task_json(sentence, model, tokenizer, schema_model=Task):
    # Prepare input
    input_text = "convert to json: " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    # Generate output
    outputs = model.generate(
        input_ids,
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    
    # Decode output
    json_string = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return json_string
