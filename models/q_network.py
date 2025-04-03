import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml

class QNetwork(nn.Module):
    def __init__(self, board_size, output_dim):
        super(QNetwork, self).__init__()
        self.board_size = board_size

        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, 256),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  
        )

        self.device = torch_directml.device()
        self.to(self.device)

        self._init_weights()

    def forward(self, x):
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
    
    model = QNetwork(board_size, output_dim)
    
    test_input = torch.randn(1, board_size, board_size)
    output = model(test_input)

    print("模型輸出形狀:", output.shape)