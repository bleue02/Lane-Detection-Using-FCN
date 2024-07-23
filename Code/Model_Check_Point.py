import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Define tensorboard writer
log_dir = '/kaggle/working/tensorboard_logs_1'
writer = SummaryWriter(log_dir)

# Define a checkpoint save function
def save_checkpoint(model, optimizer, epoch, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)

# Dummy model, optimizer, and dataloader for demonstration
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = DummyModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
dataloader = DataLoader(datasets.FakeData(transform=transforms.ToTensor()), batch_size=32, shuffle=True)

# Training loop with callbacks
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        # Log loss to tensorboard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

    # Save checkpoint
    checkpoint_filename = f'save_at_{epoch}.pth'
    save_checkpoint(model, optimizer, epoch, checkpoint_filename)

writer.close()
