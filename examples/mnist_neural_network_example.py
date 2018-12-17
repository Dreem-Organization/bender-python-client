from  __future__  import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from benderclient import Bender

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
    bender = Bender()
    bender.create_experiment(
        name='MNIST Classification',
        description='Simple image classification on handwritten digits',
        metrics=[{"metric_name": "algorithm_accuracy", "type": "reward"}],
        dataset='MNIST'
    )
    bender.create_algo(
        name='PyTorch_NN',
        hyperparameters= [
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
    return bender

if  __name__  ==  '__main__':
  # Create experiment and algo if they don't exist yet. Else, load them from the config file ./.benderconf
  bender = init_bender()
  while True:
    # Get a set of Hyperparameters to test
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
      hyperparameters=suggestion,
      results={"algorithm_accuracy": result}
    )
    print('New trial sent -----------------------------------------------------\n\n')
