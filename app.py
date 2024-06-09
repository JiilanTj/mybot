from fastapi import FastAPI, Request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()
model_name = "gpt2-finetuned"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.post("/chat/")
async def chat(request: Request):
    input_data = await request.json()
    input_text = input_data["text"]
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
