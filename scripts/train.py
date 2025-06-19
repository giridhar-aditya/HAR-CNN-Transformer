import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------- GPU Setup --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# -------- Data Augmentation (single sample: seq_len x channels) --------
def add_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def random_scaling(data, scale_range=(0.9, 1.1)):
    # data shape: (seq_len, channels)
    scales = np.random.uniform(scale_range[0], scale_range[1], (1, data.shape[1]))
    return data * scales

# -------- Dataset Classes --------
class AugmentedHAR_Dataset(Dataset):
    def __init__(self, X, y, augment=True):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            x = add_noise(x)
            x = random_scaling(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class HAR_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------- Model Definition --------
class ImprovedCNNTransformer(nn.Module):
    def __init__(self, input_channels=9, seq_len=128, num_classes=6, d_model=144, nhead=12, num_layers=6, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),  # d_model divisible by nhead
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, d_model)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# -------- Data loading utils --------
def load_signals(folder_path):
    signals = []
    axes = ['x', 'y', 'z']
    types = ['body_acc_', 'body_gyro_', 'total_acc_']
    for signal_type in types:
        for axis in axes:
            filename = os.path.join(folder_path, 'Inertial Signals', f"{signal_type}{axis}_{'train' if 'train' in folder_path else 'test'}.txt")
            signals.append(np.loadtxt(filename))
    # Result shape: (9, samples, seq_len)
    signals = np.array(signals)
    # Transpose to (samples, seq_len, 9)
    return np.transpose(signals, (1, 2, 0))

def load_labels(file_path):
    # Labels in dataset start at 1; subtract 1 for zero-based classes
    return np.loadtxt(file_path).astype(int) - 1

# -------- LR Warm-Up + Cosine Scheduler --------
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * cosine_decay for base_lr in self.base_lrs]

# -------- Hyperparameters --------
BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 3e-4
WARMUP_STEPS = 10

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total

def get_preds_labels(model, dataloader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.cpu().numpy())
            labels.append(y_batch.numpy())
    return np.concatenate(preds), np.concatenate(labels)

if __name__ == "__main__":
    # --- Load data ---
    train_folder = "UCI HAR Dataset/train"
    test_folder = "UCI HAR Dataset/test"
    X_train = load_signals(train_folder)
    y_train = load_labels(os.path.join(train_folder, "y_train.txt"))
    X_test = load_signals(test_folder)
    y_test = load_labels(os.path.join(test_folder, "y_test.txt"))

    # --- Create datasets and loaders ---
    train_dataset = AugmentedHAR_Dataset(X_train, y_train, augment=True)
    test_dataset = HAR_Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Initialize model, criterion, optimizer, scheduler ---
    model = ImprovedCNNTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = WarmUpCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS, total_steps=EPOCHS)

    # --- Training loop ---
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.8f}")

    # --- Evaluation ---
    test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")

    # --- Detailed Metrics ---
    y_pred, y_true = get_preds_labels(model, test_loader)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[
        'Walking', 'Walking Upstairs', 'Walking Downstairs',
        'Sitting', 'Standing', 'Laying'
    ]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[
                    'Walking', 'Walking Upstairs', 'Walking Downstairs',
                    'Sitting', 'Standing', 'Laying'
                ],
                yticklabels=[
                    'Walking', 'Walking Upstairs', 'Walking Downstairs',
                    'Sitting', 'Standing', 'Laying'
                ])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # --- Save model ---
    torch.save(model.state_dict(), "improved_cnn_transformer_har.pth")
    print("Model saved as improved_cnn_transformer_har2.pth")
