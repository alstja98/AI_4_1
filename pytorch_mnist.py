import numpy as np                    		# 수학 계산 관련 라이브러리
import matplotlib.pyplot as plt    		# 그래프 (및 그림) 표시를 위한 library
import torch                          		# 파이토치 관련 라이브러리
import torch.nn as nn                 		# neural network 관련 라이브러리

import torchvision.datasets as dset   		# 다양한 데이터셋 (MNIST, COCO, ...) 관련 라이브러리
import torchvision.transforms as transforms   	# 입력/출력 데이터 형태, 크기 등을 조정
from torch.utils.data import DataLoader        	# 데이터를 적절한 배치 사이즈로 load 할 수 있도록 함.

# from linearModel import My_Model
# from cnnModel import My_Model

bs = 4

class My_Model(nn.Module):
    def __init__(self):
        super().__init__()            
        self.layer = nn.Sequential(
            nn.Linear(784, 50),
            nn.ReLU(),                  
            nn.Linear(50, 100),    # 추가한 층
            nn.ReLU(),              # 추가한 활성화 함수
            nn.Linear(100, 10)      # 마지막 층
        )       
        
    def forward(self, x):            
        in_data = x.view(bs, -1)        
        out_data = self.layer(in_data)       
        return out_data


bs = 4                			# batch_size 는 대개 2^n 형태의 값으로 설정함.
learning_rate = 0.05      			# 최적 학습률은 최적화 알고리즘 및 batch_size 에 따라 달라짐.
num_epochs = 15			# 학습 반복 횟수
data_dir = "./"		# MNIST 데이터셋을 저장할 폴더

mnist_train = dset.MNIST(root=data_dir, train=True,  transform=transforms.ToTensor(), download=True)
mnist_test  = dset.MNIST(root=data_dir, train=False, transform=transforms.ToTensor(), download=True)

mnist_train = torch.utils.data.Subset(mnist_train, range(0, 600))   		# 0번부터 599번까지 600개
mnist_test  = torch.utils.data.Subset(mnist_test , range(0, 100))   		# 0번부터  99번까지 100개

train_loader = DataLoader(mnist_train, batch_size=bs, shuffle=True, drop_last=True)
test_loader  = DataLoader(mnist_test,  batch_size=bs, shuffle=True, drop_last=True)



model = My_Model()
loss_func = nn.CrossEntropyLoss()		 		# nn.CrossEntropyLoss() = [Softmax + CEL]
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   	# Stochastic Gradient Descent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")		# cuda는 GPU를 의미함.
model = model.to(device)					# GPU 사용이 가능한 경우, GPU에서 시뮬레이션 실시

def train(train_loader):
    for (data, target) in train_loader:
        (data, target) = (data.to(device), target.to(device))
        output = model(data)                        	# 순방향 전파를 통해 모델의 출력 계산
        loss = loss_func(output, target)      	# nn.CrossEntropyLoss() = [Softmax + CEL]
        optimizer.zero_grad()                    	# (모델 내의 파라미터들의) 기울기값 초기화
        loss.backward()                   		# 오차 역전파를 이용하여 기울기값 (편미분값) 계산
        optimizer.step()                   		# 파라미터 업데이트

def evaluate(test_loader):
    correct = 0			# 정답 수 초기화
    
    for (data, target) in test_loader:
        (data, target) = (data.to(device), target.to(device))
        output = model(data)        		# output.shape = (bs, 10)
        pred = torch.argmax(output, dim=1)      	# pred.shape = (bs,)
        correct += torch.sum(pred == target)	# 정답 수 업데이트

    num_test_data = len(test_loader.dataset) - (len(test_loader.dataset) % bs)
    
    test_accuracy = 100. * correct / num_test_data
    return test_accuracy

train_acc_list = [10]			# 학습 데이터에 대한 정확도 저장을 위한 list (초기값: 10%)
test_acc_list = [10]			# 시험 데이터에 대한 정확도 저장을 위한 list (초기값: 10%)

for epoch in range(1, num_epochs + 1):
    train(train_loader)			# 학습 실시
    train_accuracy = evaluate(train_loader)	# 학습 정확도 계산
    test_accuracy = evaluate(test_loader)	# 테스트 정확도 계산

    train_acc_list.append(train_accuracy.item())	# 학습 데이터에 대한 정확도 리스트 갱신
    test_acc_list.append(test_accuracy.item())	# 시험 데이터에 대한 정확도 리스트 갱신

    print(f'Epoch:{epoch:2d}   Train Acc: {train_accuracy:6.2f}%   Test Acc: {test_accuracy:5.2f}%')

x = np.arange(len(train_acc_list))

plt.plot(x, train_acc_list, label='train acc')		# 학습 데이터 정확도 출력
plt.plot(x, test_acc_list, label='test acc', linestyle='--')		# 시험 데이터 정확도 출력

plt.xlabel("epoch_num")			# x축 제목 표시
plt.ylabel("accuracy (%)")			# y축 제목 표시
plt.ylim(0, 100.0)			# y축 범위 지정
plt.legend(loc='lower right')		# 범례 표시 및 위치 지정
plt.show()