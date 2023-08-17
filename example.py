import torch 
from torch import nn
from sophia.sophia import SophiaG


#define super simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

#define a loss func
loss = nn.CrossEntropyLoss()

#optimize
optimizer = SophiaG(model.parameters(), lr=0.01, betas=(0.9, 0.999), rho=0.04,  
                    weight_decay=0.01, maximize=False, capturable=False, dynamic=True)


#generate some random data
inputs = torch.randn(1, 10)
targets = torch.randint(0, 2, (1,))

#forward pass
outputs = model(inputs)
loss = loss(outputs, targets)

#backward pass and optimization
loss.backward()
optimizer.step()

#clear the gradients for the next iteration
optimizer.zero_grad()