import torch
from torch import nn
import torch.nn.functional as f

class SampleNet(torch.nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.conv11 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv21 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 100)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.5)
        self.dropout4 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = f.relu(self.conv11(x))
        x = f.relu(self.conv12(x))
        x = f.max_pool2d(x, (2, 2))
        x = self.dropout1(x)

        x = f.relu(self.conv21(x))
        x = f.relu(self.conv22(x))
        x = f.max_pool2d(x, (2, 2))
        x = self.dropout2(x)

        x = x.view(-1, 64 * 8 * 8)
        x = f.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.dropout4(x)
        x = self.fc3(x)

        return f.log_softmax(x, dim=1)