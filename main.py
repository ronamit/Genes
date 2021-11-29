import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import random

n_genes = 8  # 1800
n_pathways = 5  # 100

class NeuralNetwork(nn.Module):
    def __init__(self, connect_mat):
        super(NeuralNetwork, self).__init__()
        self.W1 = torch.randn(n_pathways, n_genes, requires_grad=True)
        self.W2 = torch.randn(1, n_pathways, requires_grad=True)
        self.connect_mat = connect_mat

    def forward(self, x):
        hidden1 = torch.relu((torch.mul(connect_mat, self.W1)) @ x.T)
        pred = torch.sigmoid(self.W2 @ hidden1)
        return pred.T.squeeze()


connect_mat = torch.randint(0, 2, size=(n_pathways, n_genes))

# ground_truth = x[0] > 0.5

#
# pred = model(x)
# print(pred)

n_samples = 100
true_coeff = torch.randn(n_genes)
true_coeff[0] += 10.0
print('true_coeff =  ', true_coeff)
x_data = torch.zeros(n_samples, n_genes)
y_data = torch.zeros(n_samples)
for i_sample in range(n_samples):
    x = torch.rand(n_genes)
    noise = torch.randn(1) * 0.01
    y = (torch.dot(x, true_coeff) + noise) > 0
    x_data[i_sample] = x
    y_data[i_sample] = y

model = NeuralNetwork(connect_mat)

learning_rate = 1e-3
optimizer = torch.optim.Adam([model.W1, model.W2], lr=learning_rate)

n_steps = 1000
batch_size = 5
inds = list(range(n_samples))
for i_epoch in range(n_steps):
    random.shuffle(inds)
    batch_inds = inds[:batch_size]
    batch_x = x_data[batch_inds]
    batch_y = y_data[inds[:batch_size]]

    # Compute prediction and loss
    pred = model(batch_x)
    loss = F.binary_cross_entropy(pred, batch_y).sum()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


x = torch.rand(n_genes)
noise = torch.randn(1) * 0.01
xss = x.clone().detach()
y = (torch.dot(xss, true_coeff) + noise) > 0

y = torch.tensor(y, dtype=torch.float)

x.requires_grad = True
pred = model(x).unsqueeze(dim=0)
loss = F.binary_cross_entropy(pred, y)