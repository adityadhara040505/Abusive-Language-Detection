import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class AbusiveLanguageDetector(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=2, num_severity_levels=4):
        super(AbusiveLanguageDetector, self).__init__()
        try:
            self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
        except Exception as e:
            print(f"Attempting to download BERT model {model_name}...")
            self.bert = BertModel.from_pretrained(model_name)
            # Save the model locally for future offline use
            self.bert.save_pretrained(f"models/{model_name}")
            print(f"Model saved locally in models/{model_name}")
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
        # Add intermediate layers for better feature extraction
        hidden_size = self.bert.config.hidden_size
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate classifiers for abuse detection and severity
        self.abuse_classifier = nn.Linear(hidden_size // 2, num_classes)
        self.severity_classifier = nn.Linear(hidden_size // 2, num_severity_levels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use both pooled output and last hidden state
        pooled_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Combine both representations
        combined = pooled_output + last_hidden_state
        
        # Apply dropouts and intermediate layers
        x = self.dropout1(combined)
        x = self.intermediate(x)
        
        # Get both abuse and severity predictions
        abuse_logits = self.abuse_classifier(x)
        severity_logits = self.severity_classifier(x)
        
        return abuse_logits, severity_logits