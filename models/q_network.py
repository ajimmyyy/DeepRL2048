import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, board_size, output_dim):
        super(QNetwork, self).__init__()
        self.board_size = board_size

        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.ReLU()
        )

        sample_input = torch.zeros(1, 1, board_size, board_size)
        conv_output = self.conv(sample_input)
        self.flatten_size = conv_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.view(-1, 1, self.board_size, self.board_size)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
