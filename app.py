from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer
from src.model import AbusiveLanguageDetector

app = Flask(__name__)

# Load model and tokenizer globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AbusiveLanguageDetector().to(device)
model.load_state_dict(torch.load('output/best_model.pth')['model_state_dict'])
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        
        # Tokenize
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Make prediction
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            predictions = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(predictions, dim=1)
        
        result = {
            'text': text,
            'label': 'abusive' if predicted.item() == 1 else 'non-abusive',
            'confidence': float(confidence.item()),
            'probabilities': {
                'non-abusive': float(predictions[0][0].item()),
                'abusive': float(predictions[0][1].item())
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)