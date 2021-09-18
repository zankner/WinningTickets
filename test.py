import torch
from torchvision import models
import functools
from winning_ticket import ticketfy, regenerate, extract_ticket

m = models.resnet18()
ticketfy(m, 0.5)
print(m)

t = torch.randn((10, 3, 256, 256))
print(m(t).shape)

m2 = extract_ticket(m, split_rate=0.5)
print(m2(t).shape)
print(m2.fc.weight.data.shape)
# print(m.fc.weight)
# regenerate(m)
# print(m.fc.weight)