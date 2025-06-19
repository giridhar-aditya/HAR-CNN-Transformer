import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Dataset class (same as training)
class HAR_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model definition (must match your training model exactly)
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
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
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

# Load data
def load_signals(folder_path):
    signals = []
    axes = ['x', 'y', 'z']
    types = ['body_acc_', 'body_gyro_', 'total_acc_']
    for signal_type in types:
        for axis in axes:
            filename = os.path.join(folder_path, 'Inertial Signals', f"{signal_type}{axis}_test.txt")
            signals.append(np.loadtxt(filename))
    signals = np.array(signals)
    return np.transpose(signals, (1, 2, 0))

def load_labels(file_path):
    return np.loadtxt(file_path).astype(int) - 1

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

if __name__ == "__main__":
    test_folder = "UCI HAR Dataset/test"
    X_test = load_signals(test_folder)
    y_test = load_labels(os.path.join(test_folder, "y_test.txt"))

    test_dataset = HAR_Dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = ImprovedCNNTransformer().to(DEVICE)
    model.load_state_dict(torch.load("improved_cnn_transformer_har.pth", map_location=DEVICE))

    y_pred, y_true = evaluate_model(model, test_loader)

    acc = (y_pred == y_true).mean()
    print(f"Test Accuracy: {acc:.4f}\n")

    print("Classification Report:")
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
