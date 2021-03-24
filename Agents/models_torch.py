import torch.nn as nn
import torch.nn.functional as F


class PolicyDQN(nn.Module):
  def __init__(self, output_size):
    super(PolicyDQN, self).__init__()

    self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 4), stride=4)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 4), stride=4)
    self.fc = nn.Linear(2592, 256)
    self.head = nn.Linear(256, output_size)
    self.softmax = nn.Softmax()

  def forward(self, x):
    out = F.relu((self.conv1(x)))
    out = F.relu(self.conv2(out))
    out = F.relu(self.fc(out.view(out.size(0), -1)))
    out = self.softmax(self.head(out))
    return out


class ValueDQN(nn.Module):
  def __init__(self):
    super(ValueDQN, self).__init__()

    self.conv1 = nn.Conv2d(1, 16, kernel_size=8)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
    self.fc = nn.Linear(2592, 256)
    self.head = nn.Linear(256, 1)

  def forward(self, x):
    out = F.relu((self.conv1(x)))
    out = F.relu(self.conv2(out))
    out = F.relu(self.fc(out.view(out.size(0), -1)))
    out = self.head(out)
    return out