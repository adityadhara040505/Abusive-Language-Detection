import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from dotenv import load_dotenv
from data import AbusiveLanguageDataset
from model import AbusiveLanguageDetector

def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate):
    abuse_criterion = nn.CrossEntropyLoss()
    severity_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1)
    
    best_val_acc = 0
    patience = 3  # Number of epochs to wait for improvement
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            abuse_logits, severity_logits = model(input_ids, attention_mask)
            
            # Calculate losses for both tasks
            abuse_loss = abuse_criterion(abuse_logits, labels)
            severity_loss = severity_criterion(severity_logits, batch['severity'].to(device))
            
            # Combine losses
            total_loss = abuse_loss + severity_loss
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            _, predicted = torch.max(abuse_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                severities = batch['severity'].to(device)
                
                abuse_logits, severity_logits = model(input_ids, attention_mask)
                
                # Calculate validation accuracy for abuse detection
                _, predicted = torch.max(abuse_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Calculate validation accuracy for severity prediction
                _, predicted_severity = torch.max(severity_logits.data, 1)
                severity_correct = (predicted_severity == severities).sum().item()
                print(f"Severity Accuracy: {100 * severity_correct / labels.size(0):.2f}%")
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        # Save best model
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"Saving best model with validation accuracy: {val_acc:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'output/best_model.pth')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement in validation accuracy for {patience} epochs.")
            break

def main():
    # Load environment variables
    if not os.path.exists('.env'):
        print("Warning: .env file not found. Creating one with default values...")
        with open('.env', 'w') as f:
            f.write("""TRAIN_DATASET=data/train.csv
TEST_DATASET=data/test.csv
OUTPUT_DIR=output/
EPOCHS=3
BATCH_SIZE=8
LOGGING_STEPS=10
LEARNING_RATE=2e-5""")
    
    load_dotenv()
    
    # Load and validate environment variables
    TRAIN_DATASET = os.getenv('TRAIN_DATASET')
    TEST_DATASET = os.getenv('TEST_DATASET')
    
    if not TRAIN_DATASET or not TEST_DATASET:
        raise ValueError("TRAIN_DATASET and TEST_DATASET must be set in the .env file")
    
    # Convert paths to absolute paths if they're relative
    if not os.path.isabs(TRAIN_DATASET):
        TRAIN_DATASET = os.path.join(os.getcwd(), TRAIN_DATASET)
    if not os.path.isabs(TEST_DATASET):
        TEST_DATASET = os.path.join(os.getcwd(), TEST_DATASET)
    
    # Check if dataset files exist
    if not os.path.exists(TRAIN_DATASET):
        raise FileNotFoundError(f"Training dataset not found at: {TRAIN_DATASET}")
    if not os.path.exists(TEST_DATASET):
        raise FileNotFoundError(f"Test dataset not found at: {TEST_DATASET}")
    
    # Load other environment variables with default values
    EPOCHS = int(os.getenv('EPOCHS', 3))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
    
    print(f"Using configuration:")
    print(f"Training dataset: {TRAIN_DATASET}")
    print(f"Test dataset: {TEST_DATASET}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = AbusiveLanguageDataset(TRAIN_DATASET)
    test_dataset = AbusiveLanguageDataset(TEST_DATASET)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = AbusiveLanguageDetector().to(device)
    
    # Train model
    train_model(model, train_loader, test_loader, device, EPOCHS, LEARNING_RATE)

if __name__ == '__main__':
    main()