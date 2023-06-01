import numpy as np                    		
import matplotlib.pyplot as plt    		
import torch                          		
import torch.nn as nn                  	

import torchvision.datasets as dset   		
import torchvision.transforms as transforms   	
from torch.utils.data import DataLoader        

class My_Model(nn.Module):
    def __init__(self):
        super().__init__()            
        self.layer = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        
    def forward(self, x):            
        in_data = x.view(bs, -1)        
        out_data = self.layer(in_data)       
        return out_data


bs = 4           
learning_rate = 0.0005
num_epochs = 20			# Epoch 수 고정
data_dir = "./"		

mnist_train = dset.MNIST(root=data_dir, train=True,  transform=transforms.ToTensor(), download=True)
mnist_test  = dset.MNIST(root=data_dir, train=False, transform=transforms.ToTensor(), download=True)

mnist_train = torch.utils.data.Subset(mnist_train, range(0, 6000))   		
mnist_test  = torch.utils.data.Subset(mnist_test , range(0, 1000))   		


train_loader = DataLoader(mnist_train, batch_size=bs, shuffle=True, drop_last=True)
test_loader  = DataLoader(mnist_test,  batch_size=bs, shuffle=True, drop_last=True)


model = My_Model()
loss_func = nn.CrossEntropyLoss()		 	
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")		# cuda는 GPU를 의미함.		
model = model.to(device)					

def train(train_loader):
    for (data, target) in train_loader:
        (data, target) = (data.to(device), target.to(device))
        output = model(data)                        	
        loss = loss_func(output, target)      	
        optimizer.zero_grad()                    	
        loss.backward()                   		
        optimizer.step()                   		

def evaluate(test_loader):
    correct = 0			
    
    for (data, target) in test_loader:
        (data, target) = (data.to(device), target.to(device))
        output = model(data)        		
        pred = torch.argmax(output, dim=1)      	
        correct += torch.sum(pred == target)	

    num_test_data = len(test_loader.dataset) - (len(test_loader.dataset) % bs)
    
    test_accuracy = 100. * correct / num_test_data
    return test_accuracy

train_acc_list = [10]			
test_acc_list = [10]			


for epoch in range(1, num_epochs + 1):
    train(train_loader)			# 학습 실시
    train_accuracy = evaluate(train_loader)	# 학습 정확도 계산
    test_accuracy = evaluate(test_loader)	# 테스트 정확도 계산

    train_acc_list.append(train_accuracy.item())	# 학습 데이터에 대한 정확도 리스트 갱신
    test_acc_list.append(test_accuracy.item())	# 시험 데이터에 대한 정확도 리스트 갱신

    print(f'Epoch:{epoch:2d}   Train Acc: {train_accuracy:6.2f}%   Test Acc: {test_accuracy:5.2f}%')

x = np.arange(len(train_acc_list))

plt.plot(x, train_acc_list, label='train acc')	
plt.plot(x, test_acc_list, label='test acc', linestyle='--')		

plt.xlabel("epoch_num")			
plt.ylabel("accuracy (%)")			
plt.ylim(0, 100.0)			
plt.legend(loc='lower right')		
plt.show()
