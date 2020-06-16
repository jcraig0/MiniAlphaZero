import torch.nn as nn


class TicTacToeCNN(nn.Module):
    def __init__(self):
        super(TicTacToeCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 16)
        self.fc4 = nn.Linear(256, 1)

        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.drop(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.drop(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.drop(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.drop(x)
        x = self.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        y = self.fc3(x)
        y = self.soft(y).squeeze()
        z = self.fc4(x)
        z = self.sigm(z).squeeze()

        return y, z
