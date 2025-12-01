import torch, torch.nn as nn
from torch.utils.data import DataLoader


# Define a simple feedforward neural network (MLP)
class MLPClassifier(nn.Module):
  def __init__(self, d_in, n_out, layers:int=3, nodes:int=64):
    super().__init__() # Sequential block
    hidden_layers = nn.ModuleList()
    for _ in range(layers-2):
        hidden_layers.append(nn.Linear(nodes, nodes))
        hidden_layers.append(nn.ReLU())
    self.net = nn.Sequential(
      nn.Linear(d_in, nodes), # Input layer
      nn.ReLU(), # Activation
      hidden_layers,
      nn.Linear(nodes, n_out) # Output logits
    )
  def forward(self, x): # Forward pass through the network
    return self.net(x)

  def train_classifier(self, dataset, output, epochs:int=10, batch_size:int =128,
                       lr:float = 0.001):
    device = torch.device("cuda" if
         torch.cuda.is_available() else "cpu")
    model = MLPClassifier(d_in=128, n_out=output).to(device)
    opt = torch.optim.Adam(model.parameters(), lr= lr)
    loss_fn = nn.CrossEntropyLoss()

    # Hyperparameters
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)  # Move to GPU/CPU
        opt.zero_grad()             # 1. Reset gradients
        logits = model(x)           # 2. Forward pass
        loss = loss_fn(logits, y)   # 3. Compute loss
        loss.backward()             # 4. Backward pass
        opt.step()                  # 5. Update weights