# %%
import os
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%
TRAIN_PATH = "data/train"
LEADS = ['I', 'aVR','V1','V4', 'II','aVL','V2','V5','III','aVF','V3','V6']
FS = 500
TARGET_LEN = 1250

# %%
class ECGDataset(Dataset):
    def __init__(self, train_path, leads=LEADS, target_len=TARGET_LEN, transform=None):
        self.train_path = train_path
        self.records = sorted(glob.glob(os.path.join(train_path, "*.png")))
        self.leads = leads
        self.target_len = target_len
        self.transform = transform

    def __len__(self):
        return len(self.records) * len(self.leads)

    def __getitem__(self, idx):
        record_idx = idx // len(self.leads)
        lead_idx = idx % len(self.leads)

        img_path = self.records[record_idx]
        name = os.path.basename(img_path).replace(".png","")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128)) 
        img = img / 255.0
        img = np.expand_dims(img, axis=0) 

        import wfdb
        record_path = os.path.join(TRAIN_PATH, name)
        record = wfdb.rdrecord(record_path)
        gt_signal = record.p_signal.T 
        gt_lead = gt_signal[lead_idx]
        if len(gt_lead) > self.target_len:
            gt_lead = gt_lead[:self.target_len]
        else:
            gt_lead = np.pad(gt_lead, (0, self.target_len - len(gt_lead)))

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(gt_lead, dtype=torch.float32)

# %%
class ECGRegressor(nn.Module):
    def __init__(self, target_len=TARGET_LEN):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*16*16, 1024)
        self.fc2 = nn.Linear(1024, target_len)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 64x64
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 32x32
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # 16x16
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ECGDataset(TRAIN_PATH)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ECGRegressor().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for imgs, gt in dataloader:
        imgs, gt = imgs.to(device), gt.to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}  Loss={running_loss/len(dataloader):.4f}")

# %%
test_img_path = "data/test/ecg_test_0001.png"
img = cv2.imread(test_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.expand_dims(img, axis=(0,1))
img_tensor = torch.tensor(img, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    pred_signal = model(img_tensor).cpu().numpy().flatten()

print(pred_signal.shape)