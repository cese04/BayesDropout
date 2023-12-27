from scipy.special import expit
import seaborn as sns
from torch import nn
from torch import optim
import torch
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
import numpy as np

# Example parameters
DATASET_SIZE = 1000

LEARNING_RATE = 1e-2
EPOCHS = 200



# Import the utility to create the XOR dataset
from util import make_XOR

# Import the model for binary classification
from Models.classification import MLPDropBinary

# Define the color palette
cm = "coolwarm"

# X, Y = make_moons(1000, noise=0.3)

# Create the dataset
X, Y = make_XOR(1000, noise=0.7)

# Declare the tensors for input and target
X_tens = torch.from_numpy(X).float()
Y_tens = torch.from_numpy(Y).float()

# Step to create the mesh
h = 0.05

# Padding on x and y axis for the mesh
pad = 2

x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = np.c_[xx.ravel(), yy.ravel()]

Z_tens = torch.from_numpy(Z).float()

sns.scatterplot(x=X[:, 0],
            y=X[:, 1],
            c=Y,
            cmap=cm)

plt.show()

# Define the model with a custom rate por the dropout
drop_rate = 0.4
modelDrop = MLPDropBinary(layer_sizes=[2,10,10,10],
                          drop_rate=drop_rate)

# Define the optimizer, using weight_decay as a regularizer
optimizer = torch.optim.Adam(
    modelDrop.parameters(), lr=0.05, weight_decay=0.001)

# BCE as the cost function for binary classification
loss_func = torch.nn.BCEWithLogitsLoss()

# Enable the dropout
modelDrop.train()

# Set example points for sampling
data1 = torch.tensor([-2, 6]).float().unsqueeze(0)
data2 = torch.tensor([2, -4]).float().unsqueeze(0)
data3 = torch.tensor([-2, -3]).float().unsqueeze(0)

data_examples = torch.stack([data1, data2, data3])

samples, mus, sds = modelDrop.sample(data_examples)

print(samples.shape)

plt.figure()
plt.hist(x=samples)
plt.show()

