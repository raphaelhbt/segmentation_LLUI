import torch
import torch.nn as nn
import UNet_model as unet
from torchviz import make_dot
from PIL import Image
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Set parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unet.UNet3D(in_channels=1, out_channels=2).to(device="cpu")

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
batch_size = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Train the model
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

