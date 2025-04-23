import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import random
import pickle
import os
from tqdm import tqdm
import multiprocessing

print("Setting up GPU...")
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")

# Set random seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# ----------------------------
# 1. Load & Prepare Data
# ----------------------------
print("Loading data...")
df = pd.read_csv('masked_emails.csv')
X = df['masked_text']
y = df['type']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded
)

print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")
print(f"Label distribution: {np.bincount(y_encoded)}")
print(f"Classes: {le.classes_}")

# ----------------------------
# 2. Create Dataset Class
# ----------------------------
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ----------------------------
# 3. Initialize DistilBERT Tokenizer and Dataset
# ----------------------------
print("Loading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Sample a few examples to determine optimal max_length
sample_texts = X_train.sample(min(100, len(X_train)))
encodings = tokenizer(sample_texts.tolist(), truncation=True, padding=True)
lengths = [len(ids) for ids in encodings['input_ids']]
suggested_length = int(np.percentile(lengths, 95)) # Cover 95% of examples
max_length = min(512, suggested_length) # Don't exceed model's limits
print(f"Setting max sequence length to {max_length}")

# Create datasets
train_dataset = EmailDataset(X_train, y_train, tokenizer, max_length=max_length)
test_dataset = EmailDataset(X_test, y_test, tokenizer, max_length=max_length)

# ----------------------------
# 4. Initialize DistilBERT Model
# ----------------------------
print("Loading DistilBERT model...")
num_labels = len(le.classes_)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False,
)

model = model.to(device)

# Calculate class weights for imbalanced data
class_counts = np.bincount(y_train)
class_weights = torch.tensor(
    [1.0 / (count / len(y_train)) for count in class_counts],
    dtype=torch.float32
).to(device)

# ----------------------------
# 5. Define Training Functions
# ----------------------------
# Function to format time
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(),
                  lr=2e-5, # Learning rate
                  eps=1e-8 # Small value for numerical stability
                 )

# Calculate training steps
epochs = 4
total_steps = len(X_train) // 16 * epochs # Using batch size of 16

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Define loss function with class weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=True)
    
    for batch in progress_bar:
        # Clear gradients
        optimizer.zero_grad()
        
        # Get batch data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Get predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_preds, all_labels

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Update statistics
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_preds, all_labels

# Main training function
def train_and_evaluate():
    # Create dataloaders (with num_workers=0 to avoid multiprocessing issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0 # Set to 0 to avoid multiprocessing issues
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0 # Set to 0 to avoid multiprocessing issues
    )
    
    # ----------------------------
    # 6. Training Loop
    # ----------------------------
    print("Starting training...")
    training_stats = []

    for epoch in range(epochs):
        print(f"\n{'=' * 30}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 30}")
        
        # Measure time
        t0 = time.time()
        
        # Train
        train_loss, train_accuracy, _, _ = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        
        # Measure training time
        train_time = format_time(time.time() - t0)
        
        print(f"\nTraining epoch took: {train_time}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Evaluate
        t0 = time.time()
        
        val_loss, val_accuracy, val_preds, val_labels = evaluate(
            model, test_loader, device
        )
        
        # Measure validation time
        val_time = format_time(time.time() - t0)
        
        print(f"Validation took: {val_time}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Store statistics
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'train_time': train_time,
            'val_time': val_time
        })

    print("\nTraining complete!")

    # ----------------------------
    # 7. Final Evaluation
    # ----------------------------
    print("Performing final evaluation...")
    _, final_accuracy, final_preds, final_labels = evaluate(model, test_loader, device)

    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")

    print("\nClassification Report:")
    class_report = classification_report(
        final_labels, 
        final_preds, 
        target_names=le.classes_,
        digits=4
    )
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('distilbert_confusion_matrix.png')
    print("Confusion matrix saved to 'distilbert_confusion_matrix.png'")

    # Plot training & validation accuracy
    plt.figure(figsize=(12, 6))

    # Extract data
    epochs_range = [stat['epoch'] for stat in training_stats]
    train_acc = [stat['train_accuracy'] for stat in training_stats]
    val_acc = [stat['val_accuracy'] for stat in training_stats]
    train_loss = [stat['train_loss'] for stat in training_stats]
    val_loss = [stat['val_loss'] for stat in training_stats]

    # Create subplots
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('distilbert_training_metrics.png')
    print("Training metrics saved to 'distilbert_training_metrics.png'")

    # ----------------------------
    # 8. Error Analysis
    # ----------------------------
    print("Performing error analysis...")
    incorrect_indices = np.where(np.array(final_preds) != np.array(final_labels))[0]
    print(f"Total misclassifications: {len(incorrect_indices)}")

    # Find examples in the test set corresponding to incorrect predictions
    test_texts = [X_test.iloc[i] for i in range(len(X_test))]

    # Analyze a few misclassifications
    print("\nSample misclassifications:")
    for i in incorrect_indices[:5]: # Show first 5 errors
        true_label = le.inverse_transform([final_labels[i]])[0]
        pred_label = le.inverse_transform([final_preds[i]])[0]
        print(f"True: {true_label}, Predicted: {pred_label}")
        print(test_texts[i][:100] + '...' if len(test_texts[i]) > 100 else test_texts[i])
        print('-'*50)

    # ----------------------------
    # 9. Save Model
    # ----------------------------
    print("Saving model...")
    # Create output directory if it doesn't exist
    os.makedirs('distilbert_email_model', exist_ok=True)

    # Save model
    model.save_pretrained('distilbert_email_model')
    tokenizer.save_pretrained('distilbert_email_model')

    # Save label encoder and metadata
    with open('distilbert_email_model/metadata.pkl', 'wb') as f:
        pickle.dump({
            'label_encoder': le,
            'accuracy': final_accuracy,
            'classes': le.classes_,
            'max_length': max_length,
            'training_stats': training_stats
        }, f)

    print("\nModel and metadata saved to 'distilbert_email_model' directory")

    # ----------------------------
    # 10. Create Inference Function
    # ----------------------------
    def predict_email_category(text, model, tokenizer, label_encoder, max_length=512):
        """
        Predicts the category of an email text using the trained DistilBERT model.
        
        Args:
            text (str): The email text to classify
            model: The trained DistilBERT model
            tokenizer: The DistilBERT tokenizer
            label_encoder: The label encoder to convert numerical predictions to class names
            max_length (int): Maximum sequence length
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        # Ensure model is in evaluation mode
        model.eval()
        
        # Tokenize the text
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Get predicted class and confidence
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        # Convert to class name
        class_name = label_encoder.inverse_transform([pred_class])[0]
        
        return class_name, confidence

    # Example usage of the inference function
    print("\nTesting inference function with a sample email:")
    sample_text = X_test.iloc[0]
    predicted_category, confidence = predict_email_category(
        sample_text, model, tokenizer, le, max_length
    )
    true_category = le.inverse_transform([y_test[0]])[0]

    print(f"Sample email: {sample_text[:100]}...")
    print(f"True category: {true_category}")
    print(f"Predicted category: {predicted_category}")
    print(f"Confidence: {confidence:.4f}")

    print("\nTraining and evaluation complete!")
    
    return final_accuracy, class_report

if __name__ == "__main__":
    # Add freeze_support for Windows
    multiprocessing.freeze_support()
    
    # Run the training process
    final_accuracy, class_report = train_and_evaluate()
