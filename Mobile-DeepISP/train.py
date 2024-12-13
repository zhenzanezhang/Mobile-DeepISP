import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import MobileNetISP

def train_model(root_dir, epochs=10, batch_size=4, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_data_loaders(root_dir, batch_size=batch_size, num_workers=4)

    model = MobileNetISP(pretrained=True).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for raw_batch, rgb_batch in train_loader:
            raw_batch = raw_batch.to(device)
            rgb_batch = rgb_batch.to(device)

            optimizer.zero_grad()
            output = model(raw_batch)
            loss = criterion(output, rgb_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * raw_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for raw_batch, rgb_batch in val_loader:
                raw_batch = raw_batch.to(device)
                rgb_batch = rgb_batch.to(device)
                output = model(raw_batch)
                loss = criterion(output, rgb_batch)
                total_val_loss += loss.item() * raw_batch.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved.")

if __name__ == "__main__":
    train_model(root_dir='path_to_zurich_dataset')
