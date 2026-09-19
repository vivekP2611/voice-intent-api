import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import uvicorn


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "mihirparmar0913/voice-intent-qwen25"

CATEGORIES = "cancel_order, change_order, change_shipping_address, check_cancellation_fee, check_invoice, check_payment_methods, check_refund_policy, complaint, contact_customer_service, contact_human_agent, create_account, delete_account, delivery_options, delivery_period, edit_account, get_invoice, get_refund, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, review, set_up_shipping_address, switch_account, track_order, track_refund"

print("Loading model from HuggingFace Hub... This takes 2-3 minutes.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Model loaded!")

app = FastAPI(title="Customer Support Intent API", version="1.0")

class QueryInput(BaseModel):
    text: str

class IntentOutput(BaseModel):
    query: str
    predicted_intent: str
    model: str

@app.get("/health")
def health():
    return {"status": "healthy", "model": "Qwen2.5-7B-QLoRA", "categories": 27}

@app.post("/predict", response_model=IntentOutput)
def predict(input_data: QueryInput):
    prompt = f"""### Instruction:
Classify the customer support query into one of these categories: {CATEGORIES}

### Customer Query:
{input_data.text}

### Category:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    intent = response.split("### Category:")[-1].strip().split("\n")[0].strip()
    return IntentOutput(query=input_data.text, predicted_intent=intent, model="Qwen2.5-7B-QLoRA")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
