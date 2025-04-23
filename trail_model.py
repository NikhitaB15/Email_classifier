import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
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
from torch import nn
from torch.nn import CrossEntropyLoss
 
print("Setting up GPU...")
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
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Reduced max_length for roberta-large
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
# 3. Initialize RoBERTa Tokenizer and Dataset
# ----------------------------
print("Loading RoBERTa tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
 
# Sample a few examples to determine optimal max_length
sample_texts = X_train.sample(min(100, len(X_train)))
encodings = tokenizer(sample_texts.tolist(), truncation=True, padding=True)
lengths = [len(ids) for ids in encodings['input_ids']]
suggested_length = int(np.percentile(lengths, 95))
max_length = min(256, suggested_length)  # Reduced for roberta-large
print(f"Setting max sequence length to {max_length}")
 
# Create datasets
train_dataset = EmailDataset(X_train, y_train, tokenizer, max_length=max_length)
test_dataset = EmailDataset(X_test, y_test, tokenizer, max_length=max_length)
 
# ----------------------------
# 4. Custom RoBERTa Model with Class Weights
# ----------------------------
class WeightedRoberta(RobertaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights
 
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )
        logits = outputs.logits
 
        if labels is not None:
            if self.class_weights is not None:
                loss_fct = CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {'loss': loss, 'logits': logits}
       
        return {'logits': logits}
 
print("Loading RoBERTa model...")
num_labels = len(le.classes_)
 
# Calculate class weights
class_counts = np.bincount(y_train)
class_weights = torch.tensor(
    [1.0 / (count / len(y_train)) for count in class_counts],
    dtype=torch.float32
).to(device)
 
model = WeightedRoberta.from_pretrained(
    'roberta-large',
    num_labels=num_labels,
    class_weights=class_weights,
    output_attentions=False,
    output_hidden_states=False,
)
 
model = model.to(device)
 
# ----------------------------
# 5. Training Setup
# ----------------------------
# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # Reduced batch size for roberta-large
    shuffle=True,
    num_workers=0
)
 
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)
 
# Define optimizer and scheduler
optimizer = AdamW(model.parameters(),
                 lr=2e-5,
                 eps=1e-8,
                 weight_decay=0.01)
 
total_steps = len(train_loader) * 8  # 8 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
 
# Training function with early stopping
def train_model(model, train_loader, test_loader, optimizer, scheduler, device, epochs=8, patience=2):
    best_accuracy = 0
    patience_counter = 0
    training_stats = []
   
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
       
        # Training
        model.train()
        total_train_loss = 0
        train_preds = []
        train_labels = []
       
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
           
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
           
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
           
            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
       
        train_accuracy = accuracy_score(train_labels, train_preds)
        avg_train_loss = total_train_loss / len(train_loader)
       
        # Evaluation
        model.eval()
        total_eval_loss = 0
        eval_preds = []
        eval_labels = []
       
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
           
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                logits = outputs['logits']
           
            total_eval_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            eval_preds.extend(preds)
            eval_labels.extend(labels.cpu().numpy())
       
        eval_accuracy = accuracy_score(eval_labels, eval_preds)
        avg_eval_loss = total_eval_loss / len(test_loader)
       
        # Save statistics
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_eval_loss,
            'val_accuracy': eval_accuracy
        })
       
        print(f"\nTraining Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_eval_loss:.4f}")
        print(f"Validation Accuracy: {eval_accuracy:.4f}")
       
        # Early stopping
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
   
    return training_stats
 
# ----------------------------
# 6. Training
# ----------------------------
print("Starting training...")
training_stats = train_model(
    model, train_loader, test_loader, optimizer, scheduler, device
)
 
# Load the best model
model.load_state_dict(torch.load('best_model.pt'))
 
# ----------------------------
# 7. Final Evaluation
# ----------------------------
print("Performing final evaluation...")
model.eval()
total_eval_loss = 0
final_preds = []
final_labels = []
 
for batch in tqdm(test_loader, desc="Final Evaluation"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
   
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        logits = outputs['logits']
   
    total_eval_loss += loss.item()
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    final_preds.extend(preds)
    final_labels.extend(labels.cpu().numpy())
 
final_accuracy = accuracy_score(final_labels, final_preds)
avg_eval_loss = total_eval_loss / len(test_loader)
 
print(f"\nFinal Evaluation Accuracy: {final_accuracy:.4f}")
print(f"Final Evaluation Loss: {avg_eval_loss:.4f}")
 
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
plt.savefig('roberta_large_confusion_matrix.png')
print("Confusion matrix saved to 'roberta_large_confusion_matrix.png'")
 
# ----------------------------
# 8. Error Analysis
# ----------------------------
print("Performing error analysis...")
incorrect_indices = np.where(np.array(final_preds) != np.array(final_labels))[0]
print(f"Total misclassifications: {len(incorrect_indices)}")
 
test_texts = [X_test.iloc[i] for i in range(len(X_test))]
 
print("\nSample misclassifications:")
for i in incorrect_indices[:5]:
    true_label = le.inverse_transform([final_labels[i]])[0]
    pred_label = le.inverse_transform([final_preds[i]])[0]
    print(f"True: {true_label}, Predicted: {pred_label}")
    print(test_texts[i][:100] + '...' if len(test_texts[i]) > 100 else test_texts[i])
    print('-'*50)
 
# ----------------------------
# 9. Save Model
# ----------------------------
print("Saving model...")
os.makedirs('roberta_large_email_model', exist_ok=True)
 
model.save_pretrained('roberta_large_email_model')
tokenizer.save_pretrained('roberta_large_email_model')
 
with open('roberta_large_email_model/metadata.pkl', 'wb') as f:
    pickle.dump({
        'label_encoder': le,
        'accuracy': final_accuracy,
        'classes': le.classes_,
        'max_length': max_length,
        'training_stats': training_stats
    }, f)
 
print("\nModel and metadata saved to 'roberta_large_email_model' directory")
 
# ----------------------------
# 10. Inference Function
# ----------------------------
def predict_email_category(text, model, tokenizer, label_encoder, max_length=256):
    """Predicts email category using the trained RoBERTa model."""
    model.eval()
   
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
   
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
   
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
   
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    class_name = label_encoder.inverse_transform([pred_class])[0]
   
    return class_name, confidence
 
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
 