# Customer Support Intent Classifier API

Fine-tuned Qwen2.5-7B-Instruct using QLoRA for customer support intent classification.

## Model
- Base: Qwen2.5-7B-Instruct
- Fine-tuning: QLoRA (4-bit quantization)
- Dataset: 8,175 customer support queries, 27 intent categories
- Accuracy: 100% on held-out test samples
- Model Hub: https://huggingface.co/mihirparmar0913/voice-intent-qwen25

## Endpoints
- GET /health - Health check
- POST /predict - Predict intent from customer query

## Example
curl -X POST "https://your-api.onrender.com/predict" -H "Content-Type: application/json" -d '{"text": "I want to cancel my order"}'
# voice-intent-api
