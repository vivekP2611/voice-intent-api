import os
os.environ["PEFT_NO_TORCHAO"] = "1"

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "mihirparmar0913/voice-intent-qwen25"

CATEGORIES = "cancel_order, change_order, change_shipping_address, check_cancellation_fee, check_invoice, check_payment_methods, check_refund_policy, complaint, contact_customer_service, contact_human_agent, create_account, delete_account, delivery_options, delivery_period, edit_account, get_invoice, get_refund, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, review, set_up_shipping_address, switch_account, track_order, track_refund"

print("Loading model... This takes 3-4 minutes.")

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
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
else:
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to("cpu")

model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print(f"Model loaded on {device}!")

def predict_intent(query):
    prompt = f"""### Instruction:
Classify the customer support query into one of these categories: {CATEGORIES}

### Customer Query:
{query}

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
    return intent

demo = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(
        label="Enter Customer Support Query",
        lines=2,
        placeholder="e.g., I want to cancel my order"
    ),
    outputs=gr.Textbox(label="Predicted Intent"),
    title="Customer Support Intent Classifier",
    description="Fine-tuned Qwen2.5-7B (QLoRA) on 8K+ support queries. 27 intent categories.",
    examples=[
        "I want to cancel my order",
        "My payment failed but money was deducted",
        "How do I track my delivery?",
        "I need to change my shipping address",
        "Can I get a refund for my last purchase?",
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
