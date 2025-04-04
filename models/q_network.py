import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml

class QNetwork(nn.Module):
    def __init__(self, board_size, output_dim):
        super(QNetwork, self).__init__()
        self.board_size = board_size

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

        self._init_weights()

    def forward(self, x):
        x = torch.log2(x + 1)
        x = x.view(-1, 1, self.board_size, self.board_size)
        x = self.conv(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def _init_weights(self):
        """ 初始化權重 """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    board_size = 4
    output_dim = 4
    device = torch.device("cpu")
    
    model = QNetwork(board_size, output_dim).to(device)
    
    test_input = torch.randn(1, board_size, board_size).to(device)
    output = model(test_input)

    print("模型輸出形狀:", output.shape)