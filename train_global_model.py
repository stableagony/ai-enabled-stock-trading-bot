import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Parameters
SEQUENCE_LENGTH = 60
FUTURE_PERIODS = 15
BATCH_SIZE = 256
NUM_EPOCHS = 3
HIDDEN_SIZE = 128  # Larger model
NUM_LAYERS = 3     # Deeper model
LEARNING_RATE = 0.001

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TradingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def prepare_data_from_file(file_path):
    df = pd.read_parquet(file_path).iloc[-20000:].copy()
    # Target: 1 if price goes UP in future periods
    df['Target'] = (df['close'].shift(-FUTURE_PERIODS) > df['close']).astype(int)
    df.dropna(inplace=True)
    
    features = ['open', 'high', 'low', 'close', 'volume']
    data_values = df[features].values
    target_values = df['Target'].values
    
    X, y = [], []
    for i in range(len(df) - SEQUENCE_LENGTH):
        X.append(data_values[i : i + SEQUENCE_LENGTH])
        y.append(target_values[i + SEQUENCE_LENGTH - 1])
        
    return X, y

if __name__ == '__main__':
    # Select 10 random files to create a generalized, larger model without running out of memory
    all_files = glob.glob('normalized_data/*.parquet')
    selected_files = random.sample(all_files, min(5, len(all_files)))
    
    print(f"Training larger generalized model on {len(selected_files)} stocks...")
    
    all_X, all_y = [], []
    for file in selected_files:
        print(f"Processing {os.path.basename(file)}...")
        X, y = prepare_data_from_file(file)
        all_X.extend(X)
        all_y.extend(y)
        
    all_X = np.array(all_X)
    all_y = np.array(all_y)
    
    print(f"Total training sequences: {len(all_X)}")
    
    dataset = StockDataset(all_X, all_y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = TradingLSTM(input_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")
        
    # Save the model
    torch.save(model.state_dict(), 'global_lstm.pth')
    print("Saved trained model to global_lstm.pth")
