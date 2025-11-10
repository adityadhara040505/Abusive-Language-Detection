import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer

class AbusiveLanguageDataset(Dataset):
    def __init__(self, data_path, tokenizer_name='bert-base-uncased', max_length=128):
        try:
            self.data = pd.read_csv(data_path)
        except Exception as e:
            raise RuntimeError(f"Error reading dataset from {data_path}: {str(e)}")
            
        # Validate required columns
        required_columns = {'text', 'label', 'severity'}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}")
            
        # Validate data types and values
        if not all(self.data['label'].isin([0, 1])):
            raise ValueError("Labels must be binary (0 or 1)")
        if not all(self.data['severity'].isin([0, 1, 2, 3])):
            raise ValueError("Severity must be between 0 and 3")
            
        try:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise RuntimeError(f"Error loading tokenizer {tokenizer_name}: {str(e)}")
            
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = int(self.data.iloc[idx]['label'])
        severity = int(self.data.iloc[idx]['severity'])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label,
            'severity': severity
        }