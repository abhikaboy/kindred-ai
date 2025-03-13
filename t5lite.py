   

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# tokenize input 
inputText = "Buy a new whisk"


outputs = model.generate(
        input_ids, 
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)