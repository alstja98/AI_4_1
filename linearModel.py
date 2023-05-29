import torch                          		# 파이토치 관련 라이브러리
import torch.nn as nn      


bs = 4

class My_Model(nn.Module):
    def __init__(self):
        super().__init__()			# parent class 인 nn.Module의 생성자/초기화 함수를 상속함.
        self.layer = nn.Sequential(
            nn.Linear(784, 50),
            nn.ReLU(),			# nn.Sigmoid() 함수를 사용할 수도 있음.
            nn.Linear(50, 10)
        )       
        
    def forward(self, x):			# x.shape = (bs, 1, 28, 28)
        in_data = x.view(bs, -1)		# in_data.shape = (bs, 784)
        out_data = self.layer(in_data)		# out_data.shape = (bs, 10)
        return out_data