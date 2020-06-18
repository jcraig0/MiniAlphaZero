import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, game):
        super(CNN, self).__init__()

        # The padding cancels out the kernel size shrinking the output
        self.conv1 = nn.Conv2d(1, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)

        if game == 'tic-tac-toe-3':
            fc_values = (2304, 576, 144, 9)
        elif game == 'tic-tac-toe-4':
            fc_values = (4096, 1024, 256, 16)
        else:
            fc_values = (10752, 2688, 672, 7)

        self.fc1 = nn.Linear(fc_values[0], fc_values[1])
        self.fc2 = nn.Linear(fc_values[1], fc_values[2])
        self.fc3 = nn.Linear(fc_values[2], fc_values[3])
        self.fc4 = nn.Linear(fc_values[2], 1)

        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        self.sigm = nn.Sigmoid()

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

        # Policy head (vector of floats between 0 and 1)
        y = self.fc3(x)
        y = self.soft(y).squeeze()
        # Value head (scalar float between 0 and 1)
        z = self.fc4(x)
        z = self.sigm(z).squeeze()

        return y, z
