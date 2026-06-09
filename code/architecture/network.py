import torch
import torch.nn as nn


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.3):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc5(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)
        if self.numRes > 0:
            x = self.resBlocks(x)

        return x



class DimReduction1(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction1, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res
        # 残差投影：当输入维度与输出维度不同时，需要投影到相同维度
        self.shortcut = nn.Linear(n_channels, m_dim, bias=False) if n_channels != m_dim else nn.Identity()

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        x_ = self.shortcut(x)
        x = self.fc1(x)
        x = self.relu1(x + x_)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x



