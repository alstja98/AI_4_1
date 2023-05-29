import torch                          		# 파이토치 관련 라이브러리
import torch.nn as nn      

bs = 4

class My_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=5),      # [bs, 1, 28, 28] -> [bs, 16, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                          	# [bs, 16, 24, 24] -> [bs, 16, 12, 12]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),   	# [bs, 16, 12, 12] -> [bs, 32, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                          	# [bs, 32, 8, 8] -> [bs, 32, 4, 4]
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(32*4*4, 10),                                          		# [bs, 32*4*4] -> [bs, 10]
        )
        
    def forward(self, x):                           	# x.shape = (bs, 1, 28, 28)
        out_data = self.conv_layer(x)               	# out_data.shape = (bs, 32, 4, 4)
        out_data = out_data.view(bs, -1)	# out_data.shape = (bs, 32*4*4) = (bs, 512)
        out_data = self.fc_layer(out_data)          	# out_data.shape = (bs, 10)
        return out_data
