import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set random seed and device
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
with open('dataset.json', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file.readlines()]
data = random.sample(data, 1000)  # Randomly sampling data points
titles = [d['title'] for d in data]
classes = sorted(set(d['class'] for d in data))
class_to_index = {cls: i for i, cls in enumerate(classes)}
labels = [class_to_index[d['class']] for d in data]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
encoded_data = tokenizer(titles, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Dataset class
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Preparing dataset and dataloaders
dataset = NewsDataset(encoded_data, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(classes)).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training and evaluation
train_losses = []
val_accuracies = []



model.train()
for epoch in range(3):  # Adjust the number of epochs as necessary
    print(f"Epoch {epoch+1}")
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        """
        # Evaluate the model every ten steps
        if step % 10 == 0:
            model.eval()
            total, correct = 0, 0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)

            val_accuracies.append(correct / total)
            model.train()"""

all_preds = []
all_true = []

model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_true.extend(batch['labels'].cpu().numpy())

# Compute the confusion matrix
cm = confusion_matrix(all_true, all_preds, labels=range(len(classes)))

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()