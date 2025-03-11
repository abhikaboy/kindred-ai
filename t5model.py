from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np

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

# Split dataset (80% train, 10% validation, 10% test)
train_test = hf_dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)

dataset_splits = {
    "train": train_test["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"]
}

print(f"Train: {len(dataset_splits['train'])} examples")
print(f"Validation: {len(dataset_splits['validation'])} examples")
print(f"Test: {len(dataset_splits['test'])} examples")


def preprocess(entries):
    model_inputs = tokenizer(
        # Important to tell the model specifically what transformation to preform
        ["convert to json: " + ex for ex in entries["input"]],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    # Tokenize outputs
    labels = tokenizer(
        entries["output"],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    
    model_inputs["labels"] = labels

    # return after processing 
    return model_inputs

# Process each split with a progress bar
tokenized_datasets = {}
for split_name, split_dataset in dataset_splits.items():
    print(f"Processing {split_name} split...")
    tokenized_datasets[split_name] = split_dataset.map(
        preprocess,
        batched=True,
        batch_size=8,
        remove_columns=["input", "output"],
        desc=f"Tokenizing {split_name}"
    )
    # Show an example from each processed split
    print(f"Example from {split_name}:")
    # example_features = {k: v[0].tolist() for k, v in tokenized_datasets[split_name][0].items()}
    # for k, v in example_features.items():
    #     print(f"  {k}: {v[:10]}... (length: {len(v)})")



# Model 
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Print model architecture details
print("\nModel Information:")
print(f"Model type: {model.__class__.__name__}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Look at some key components of the model
print("\nModel Architecture Highlights:")
print(f"Encoder: {model.encoder.__class__.__name__}")
print(f"Decoder: {model.decoder.__class__.__name__}")
print(f"LM Head shape: {model.lm_head.weight.shape}")



def compute_json_accuracy(eval_preds):
    predictions, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Track different types of accuracy
    exact_match = 0
    valid_json = 0
    field_accuracy = {
        "priority": 0,
        "content": 0,
        "difficulty": 0,
        "recurring": 0,
        # Add other fields
    }
    total_count = len(decoded_preds)
    
    for pred, label in zip(decoded_preds, decoded_labels):
        # Check for exact match
        if pred == label:
            exact_match += 1
        
        # Check for valid JSON
        try:
            pred_json = json.loads(pred)
            label_json = json.loads(label)
            valid_json += 1
            
            # Check field-level accuracy
            for field in field_accuracy.keys():
                if field in pred_json and field in label_json:
                    if pred_json[field] == label_json[field]:
                        field_accuracy[field] += 1
        except:
            # Invalid JSON prediction
            print("Failed JSON Validation")
            pass
    
    # Calculate accuracies
    results = {
        "exact_match": exact_match / total_count,
        "valid_json_rate": valid_json / total_count,
    }
    
    # Add field-level accuracies
    for field, correct in field_accuracy.items():
        results[f"{field}_accuracy"] = correct / total_count
    
    return results



# Set up training arguments

# Set up TensorBoard
log_dir = f"runs/t5_json_converter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
tb_writer = SummaryWriter(log_dir)
print(f"TensorBoard logs will be saved to: {log_dir}")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_dir=log_dir,
    logging_steps=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="valid_json_rate",
    greater_is_better=True,
    save_total_limit=3,
    report_to="tensorboard",
)

# Set up early stopping
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=30,
    early_stopping_threshold=0.05
)

# PyTorch Trainer with custom logging
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        # Standard PyTorch forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Log detailed loss information
        if self.state.global_step % 10 == 0:
            self.log({"detailed_loss": loss.item()})
            
            # You can add more custom PyTorch-specific logging here
            for name, param in model.named_parameters():
                if param.grad is not None and self.state.global_step % 100 == 0:
                    self.log({f"grad_norm/{name}": param.grad.norm().item()})
        
        return (loss, outputs) if return_outputs else loss

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_json_accuracy,
    callbacks=[early_stopping_callback]
)

# PyTorch-specific: Add hooks to see what's happening inside the model
activation = {}

def get_activation(name):
    def hook(model, input, output):
        # Handle tuple outputs by storing the first element
        if isinstance(output, tuple):
            activation[name] = output[0].detach()
        else:
            activation[name] = output.detach()
    return hook

# Add hooks to key parts of the model
model.encoder.block[0].layer[0].SelfAttention.register_forward_hook(get_activation('self_attention'))

# Train the model
print("\nStarting training...")
train_result = trainer.train()

# Print training results
print("\nTraining completed!")
print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
print(f"Training loss: {train_result.metrics['train_loss']:.4f}")




# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(tokenized_datasets["test"])
print("Test results:")
for metric_name, value in test_results.items():
    print(f"  {metric_name}: {value:.4f}")

# Detailed error analysis
print("\nDetailed error analysis:")
model.eval()
errors = []

for i, example in enumerate(dataset_splits["test"]):
    # Get input and expected output
    input_text = "convert to json: " + example["input"]
    expected_output = example["output"]
    
    # Get model prediction
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids, 
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Check for errors
    try:
        pred_json = json.loads(predicted_text)
        expected_json = json.loads(expected_output)
        
        # Check for field errors
        field_errors = []
        for key in expected_json:
            if key not in pred_json or pred_json[key] != expected_json[key]:
                field_errors.append(key)
        
        if field_errors:
            errors.append({
                "input": example["input"],
                "predicted": pred_json,
                "expected": expected_json,
                "field_errors": field_errors
            })
    except json.JSONDecodeError:
        errors.append({
            "input": example["input"],
            "predicted": predicted_text,
            "expected": expected_json,
            "error": "Invalid JSON"
        })

# Print some error examples
if errors:
    print(f"Found {len(errors)} errors out of {len(dataset_splits['test'])} examples")
    print("\nSample errors:")
    for i, error in enumerate(errors[:3]):
        print(f"\nError {i+1}:")
        print(f"Input: {error['input']}")
        print(f"Predicted: {error['predicted']}")
        print(f"Expected: {error['expected']}")
        if "field_errors" in error:
            print(f"Field errors: {error['field_errors']}")
        else:
            print(f"Error: {error['error']}")
else:
    print("No errors found!")


# Save model and tokenizer
print("\nSaving model and tokenizer...")
model_path = "./results/final-model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")

# Create inference function
def sentence_to_task_json(sentence, model, tokenizer):
    # Prepare input
    input_text = "convert to json: " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    # Generate with detailed settings
    outputs = model.generate(
        input_ids,
        max_length=512,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0
    )
    
    # Decode output
    json_string = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        # Parse JSON
        json_obj = json.loads(json_string)
        
        # Validate against schema
        validated_task = Task(**json_obj)
        return validated_task.dict()
    except Exception as e:
        print(f"Error parsing generated JSON: {e}")
        print(f"Generated string: {json_string}")
        return None




# Test the inference pipeline on new examples
print("\nTesting inference pipeline...")
test_sentences = [
    "Read a chapter of a book before bed at 9:00 PM.",
    "Water plants at 6:30 PM every Monday and Thursday.",
    "Attend team meeting at 2:00 PM tomorrow."
]

for sentence in test_sentences:
    print(f"\nInput: {sentence}")
    result = sentence_to_task_json(sentence, model, tokenizer)
    if result:
        print(f"Output JSON:")
        print(json.dumps(result, indent=2))
    else:
        print("Failed to convert to valid JSON")

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
