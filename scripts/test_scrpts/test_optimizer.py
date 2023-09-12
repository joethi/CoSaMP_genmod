import torch
import torch.nn as nn
torch.manual_seed(1)
x = torch.arange(-5,5,0.1).view(-1,1)
x = torch.Tensor([5]).view(-1,1)
noise = 0.4 * torch.randn(x.size())
y = 10*x -1 + noise
import pdb; pdb.set_trace()
torch.manual_seed(0)
model = nn.Linear(1,1)
criterion = torch.nn.MSELoss() 
#model = nn.Sequential(nn.Linear(1,1),nn.Linear(1,1),nn.Linear(1,1))
print(list(model.parameters()))
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
model.state_dict()['weight'][0] = 10.0
model.state_dict()['bias'][0] = -1.0
#model.state_dict()['weight'] = torch.full_like(model.state_dict()['weight'], 3.0)
#model.state_dict()['weight'].fill_(3.0)
print(list(model.parameters()))
yhat = model(x)
loss = criterion(y,yhat)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(list(model.parameters()))
#model.state_dict()['weight'] = 10
#model.state_dict()['bias'] = -1
#optimizer_1 = torch.optim.SGD(model.parameters(),lr=0.01)

#optimizer.zero_grad()
#optimizer.step()

import pdb; pdb.set_trace()

