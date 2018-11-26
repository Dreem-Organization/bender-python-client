# Bender Client for Python

> :bar_chart: You can see a quick online demo of the project [HERE](https://bender.dreem.com/demo).

> :warning: The full **DOCUMENTATION** on bender-python-client can be found [HERE](https://bender-optimizer.readthedocs.io/en/latest/documentation/python.html).

## Setup

 1. Create an account for free at [bender.dreem.com](https://bender.dreem.com)
 2. Install bender in your Python environment with ``` pip install bender-client ```

## Usage Example

> Let's use the famous MNIST example where we try to recognize handwritten digits in images.

The code of the algorithm using [PyTorch](https://pytorch.org/) is the following :

> To use this example, do not forget to ``` pip install numpy torch torchvision ``` .

```python
from  __future__  import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
  def __init__(self, dropout=True, activation="relu", kernel_size=5, conv_depth=10, linear_depth=50):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, conv_depth, kernel_size=kernel_size)
    self.conv2 = nn.Conv2d(conv_depth, 20, kernel_size=kernel_size)
    self.conv2_drop = nn.Dropout2d() if dropout is  True  else  lambda  x: x
    self.fc1 = nn.Linear(320, linear_depth)
    self.fc2 = nn.Linear(linear_depth, 10)
    self.activation =  getattr(F, activation)

  def forward(self, x):
    x = self.activation(F.max_pool2d(self.conv1(x), 2))
    x = self.activation(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = self.activation(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x =  self.fc2(x)
    return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in  enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 ==  0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx *  len(data), len(train_loader.dataset),
        100. * batch_idx /  len(train_loader), loss.item()))

def test(model, device, test_loader):
  model.eval()
  test_loss =  0
  correct =  0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
  test_loss /=  len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct /  len(test_loader.dataset)))
  return (correct / len(test_loader.dataset))

def run(epochs=3, lr=0.01, momentum=0.5, dropout=True, activation="relu", kernel_size=5, conv_depth=10, linear_depth=50):
  torch.manual_seed(1)
  device = torch.device("cpu")
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
      transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=32,
    shuffle=True,
  )
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1000,
    shuffle=True,
  )

  model = Net(dropout, activation).to(device)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  accuracy = 0
  for epoch in  range(1, int(epochs) +  1):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
  return accuracy

if  __name__  ==  '__main__':
  # HYPER PARAMETERS (That's what bender is interested in)
  # Here we select them on our own in an arbitrary way
  hyper_parameters = {
    "kernel_size": 5,
    "epochs": 3,
    "lr": 0.05,
    "momentum": 0.2,
    "dropout": True,
    "activation": "relu",
    "conv_depth": 10,
    "linear_depth": 50,
  }
  run(
    epochs=hyper_parameters["epochs"],
    lr=hyper_parameters["lr"],
    momentum=hyper_parameters["momentum"],
    dropout=hyper_parameters["dropout"],
    activation=hyper_parameters["activation"],
    kernel_size=hyper_parameters["kernel_size"],
    conv_depth=hyper_parameters["conv_depth"],
    linear_depth=hyper_parameters["linear_depth"],
  )
```

***Now let's plug Bender into It !***

 1. **Importing Bender**
```python
from bender import Bender
bender = Bender()
```

> This will ask for your email and password. The client will use these to login and retrieve a TOKEN.
> This TOKEN is personal, it should not be shared, it will be stored in your home folder as "`.bender_token`", and you will not be asked for your login/password again until it expires.
>:warning: Again, your TOKEN is personal. You should not give it or add it to any public repository :warning:

 2. **Create an Experiment**
> *An experiment is related to the problem you are trying to solve, here : MNIST classification*
```python
bender.new_experiment(
	name='MNIST Classification',
	description='Simple image classification on handwritten digits',
	metrics=[
		{
			"metric_name": "algorithm_accuracy", # It's just a name and there can be multiple watched metrics.
			"type": "reward", # The type can either be "reward" or "loss" depending on if you want to maximize or minimize it.
		}
	],
	dataset='MNIST'
)
```
3. **Create an Algo**
> *An algo is simply corresponding to ONE solution to an Experiment problem : here it's as we saw a Neural Net with PyTorch*
```python
bender.new_algo(
	name='PyTorch_NN',
	# The parameters below are actually the Hyper-Parameters of your algo described in a list
	parameters= [
    {
      "name": 'kernel_size',
      "category": "categorical",
      "search_space": {
        "values": [3, 5, 7],
      },
    },
    {
      "name": 'conv_depth',
      "category": "uniform",
      "search_space": {
        "low": 1,
        "high": 100,
        "step": 1,
      },
    },
    {
      "name": 'linear_depth',
      "category": "uniform",
      "search_space": {
        "low": 1,
        "high": 100,
        "step": 1,
      },
    },
    {
      "name": 'epochs',
      "category": "uniform",
      "search_space": {
        "low": 1,
        "high": 4,
        "step": 1,
      },
    },
    {
      "name": 'lr',
      "category": "loguniform",
      "search_space": {
        "low": 1e-5,
        "high": 1e-1,
        "step": 1e-6,
      },
    },
    {
      "name": 'momentum',
      "category": "uniform",
      "search_space": {
        "low": 0,
        "high": 1,
        "step": 0.05,
      },
    },
    {
      "name": 'dropout',
      "category": "categorical",
      "search_space": {
        "values": [True, False],
      },
    },
    {
      "name": 'activation',
      "category": "categorical",
      "search_space": {
        "values": ["relu", "softmax", "sigmoid", "tanh"],
      },
    },
  ]
)
```
 
4. **Get an Hyper Parameters Set suggestion from Bender**

> *The whole goal of what we did up there is to use Bender to get a new set of Hyper Parameters to try according to the settings of your Experiment and Algo.*
 
```python
suggestion = bender.suggest() 

# suggestion would for example contain something like :
{
  "kernel_size": 5,
  "epochs": 3,
  "lr": 0.05,
  "momentum": 0.2,
  "dropout": True,
  "activation": "tanh",
  "conv_depth": 10,
  "linear_depth": 50,
}
```

5. **Feed a Trial to Bender**

> *A Trial is simply an attempt of you trying a Hyper Parameters Set with your algorithm associated with the result metrics obtained. If you want bender to improve over time, feed him every trial you make.*

```python
bender.new_trial(
	parameters={
    "kernel_size": 5,
		"epochs": 3,
		"lr": 0.05,
		"momentum": 0.2,
		"dropout": True,
		"activation": "tanh",
    "conv_depth": 10,
    "linear_depth": 50,
	},
	results={
		"algorithm_accuracy": 0.7, # We put an arbitrary value here just for the example.
	}
)
```
6. **The full code put together**

> *Psssssst... The magic starts at line 443... ;)*

> To use this example, do not forget to ``` pip install numpy torch torchvision bender-client ``` .

```python
from  __future__  import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from benderclient import Bender
bender = Bender()

class Net(nn.Module):
  def __init__(self, dropout=True, activation="relu", kernel_size=5, conv_depth=10, linear_depth=50):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, conv_depth, kernel_size=kernel_size)
    self.conv2 = nn.Conv2d(conv_depth, 20, kernel_size=kernel_size)
    self.conv2_drop = nn.Dropout2d() if dropout is  True  else  lambda  x: x
    self.fc1 = nn.Linear(320, linear_depth)
    self.fc2 = nn.Linear(linear_depth, 10)
    self.activation =  getattr(F, activation)

  def forward(self, x):
    x = self.activation(F.max_pool2d(self.conv1(x), 2))
    x = self.activation(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = self.activation(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x =  self.fc2(x)
    return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in  enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 ==  0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx *  len(data), len(train_loader.dataset),
        100. * batch_idx /  len(train_loader), loss.item()))

def test(model, device, test_loader):
  model.eval()
  test_loss =  0
  correct =  0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
  test_loss /=  len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct /  len(test_loader.dataset)))
  return (correct / len(test_loader.dataset))

def run(epochs=3, lr=0.01, momentum=0.5, dropout=True, activation="relu", kernel_size=5, conv_depth=10, linear_depth=50):
  torch.manual_seed(1)
  device = torch.device("cpu")
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
      transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=32,
    shuffle=True,
  )
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1000,
    shuffle=True,
  )

  model = Net(dropout, activation).to(device)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  accuracy = 0
  for epoch in  range(1, int(epochs) +  1):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
  return accuracy

def init_bender():
    global bender
    bender.create_experiment(
        name='MNIST Classification',
        description='Simple image classification on handwritten digits',
        metrics=[{"metric_name": "algorithm_accuracy", "type": "reward"}],
        dataset='MNIST'
    )
    bender.create_algo(
        name='PyTorch_NN',
        hyper_parameters= [
            {
                "name": 'kernel_size',
                "category": "categorical",
                "search_space": {
                    "values": [3, 5, 7],
                },
            },
            {
                "name": 'conv_depth',
                "category": "uniform",
                "search_space": {
                    "low": 1,
                    "high": 100,
                    "step": 1,
                },
            },
            {
                "name": 'linear_depth',
                "category": "uniform",
                "search_space": {
                    "low": 1,
                    "high": 100,
                    "step": 1,
                },
            },
            {
                "name": 'epochs',
                "category": "uniform",
                "search_space": {
                    "low": 1,
                    "high": 4,
                    "step": 1,
                },
            },
            {
                "name": 'lr',
                "category": "loguniform",
                "search_space": {
                    "low": 1e-5,
                    "high": 1e-1,
                    "step": 1e-6,
                },
            },
            {
                "name": 'momentum',
                "category": "uniform",
                "search_space": {
                    "low": 0,
                    "high": 1,
                    "step": 0.05,
                },
            },
            {
                "name": 'dropout',
                "category": "categorical",
                "search_space": {
                    "values": [True, False],
                },
            },
            {
                "name": 'activation',
                "category": "categorical",
                "search_space": {
                    "values": ["relu", "softmax", "sigmoid", "tanh"],
                },
            },
        ]
    )

if  __name__  ==  '__main__':
  # Create experiment and algo if they don't exist yet. Else, load them from the config file ./.benderconf
  init_bender()
  while True:
    # Get a set of Hyper Parameters to test
    suggestion = bender.suggest(metric="algorithm_accuracy")
    # Get algo result with them
    result = run(
      epochs=suggestion["epochs"],
      lr=suggestion["lr"],
      momentum=suggestion["momentum"],
      dropout=suggestion["dropout"],
      activation=suggestion["activation"],
      kernel_size=suggestion["kernel_size"],
      conv_depth=suggestion["conv_depth"],
      linear_depth=suggestion["linear_depth"],
    )
    # Feed Bender a Trial, AKA => suggestion + result
    bender.create_trial(
      hyper_parameters=suggestion,
      results={"algorithm_accuracy": result}
    )
    print('New trial sent -----------------------------------------------------\n\n')
```