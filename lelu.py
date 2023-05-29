import torch
import torch.nn as nn

a = torch.tensor(-1.)
print(a)
layer_1 = nn.ReLU()
b = layer_1(a)
print(b)
print(a) #tensor(-1.)

layer_2 = nn.ReLU(inplace=True)
b = layer_2(a)
print(b)
print(a) #tensor(0.)

my_layer = nn.ReLU()
input = torch.randn(4)
output = my_layer(input)
print(input)
print(output)