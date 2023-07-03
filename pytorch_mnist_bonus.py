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
num_epochs = 20
data_dir = "./"		

mnist_train = dset.MNIST(root=data_dir, train=True,  transform=transforms.ToTensor(), download=True)
mnist_test  = dset.MNIST(root=data_dir, train=False, transform=transforms.ToTensor(), download=True)

mnist_train = torch.utils.data.Subset(mnist_train, range(0, 6000))   		
mnist_test  = torch.utils.data.Subset(mnist_test , range(0, 1000))   		


train_loader = DataLoader(mnist_train, batch_size=bs, shuffle=True, drop_last=True)
test_loader  = DataLoader(mnist_test,  batch_size=bs, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")		

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


learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]  
train_accuracies = {lr: [] for lr in learning_rates}
test_accuracies = {lr: [] for lr in learning_rates}

for learning_rate in learning_rates:
    model = My_Model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()  
    train_acc_list = [10]
    test_acc_list = [10]

    for epoch in range(1, num_epochs + 1):
        train(train_loader)
        train_accuracy = evaluate(train_loader) 
        test_accuracy = evaluate(test_loader)
        train_acc_list.append(train_accuracy.item())
        test_acc_list.append(test_accuracy.item())
        print(f'Learning rate: {learning_rate}  Epoch:{epoch:2d}   Train Acc: {train_accuracy:6.2f}%   Test Acc: {test_accuracy:5.2f}%')

    train_accuracies[learning_rate] = train_acc_list
    test_accuracies[learning_rate] = test_acc_list

for lr in learning_rates:
    x = np.arange(len(train_accuracies[lr]))
    plt.plot(x, train_accuracies[lr], label=f'train acc, lr={lr}')
    plt.plot(x, test_accuracies[lr], label=f'test acc, lr={lr}', linestyle='--')

plt.xlabel("epoch_num")
plt.ylabel("accuracy (%)")
plt.ylim(0, 100.0)
plt.legend(loc='lower right')
plt.show()
