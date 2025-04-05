import torch
import torch.nn as nn
import torch_directml

class DuelingQNetwork(nn.Module):
    def __init__(self, board_size, output_dim):
        super(DuelingQNetwork, self).__init__()
        self.board_size = board_size

        # 卷積層
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

        # 價值分支 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # 優勢分支 A(s, a)
        self.advantage_stream = nn.Sequential(
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

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

    def _init_weights(self):
        """ 初始化權重 """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# 測試
if __name__ == "__main__":
    board_size = 4
    output_dim = 4
    device = torch.device("cpu")

    model = DuelingQNetwork(board_size, output_dim).to(device)
    test_input = torch.randn(1, board_size, board_size).to(device)
    output = model(test_input)

    print("模型輸出形狀:", output.shape)
