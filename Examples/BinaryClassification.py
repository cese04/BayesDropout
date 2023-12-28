from scipy.special import expit
import seaborn as sns
from torch import nn
from torch import optim
import torch
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
import numpy as np

# Example parameters
DATASET = 'XOR'
DATASET_SIZE = 1000

LEARNING_RATE = 1e-2
EPOCHS = 1000


# Import the utility to create the XOR dataset
from util import make_XOR

# Import the model for binary classification
from Models.classification import MLPDropBinary

# Define the color palette
cm = "coolwarm"


X = np.array([])
Y = np.array([])
# Create the dataset
if DATASET == "XOR":
    X, Y = make_XOR(1000, noise=0.7)
elif DATASET == "moons":
    X, Y = make_moons(1000, noise=0.3)


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

#### Displaying the initial state of the model

# Define the model with a custom rate por the dropout
drop_rate = 0.4
modelDrop = MLPDropBinary(layer_sizes=[2,10,10,10],
                          drop_rate=drop_rate)

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
for i in range(len(samples)):
    plt.hist(x=samples[i], bins=20, range=(0,1), alpha=0.6)
    plt.axvline(mus[i].numpy())
plt.show()

##### Training the model

# Define the optimizer, using weight_decay as a regularizer
optimizer = torch.optim.Adam(
    modelDrop.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.001)

# BCE as the cost function for binary classification
loss_func = torch.nn.BCEWithLogitsLoss()


modelDrop.train()

for e in range(EPOCHS):

    out = modelDrop(X_tens)

    loss = loss_func(out.view(len(Y_tens)), Y_tens)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


##### Display a meshgrid with the predictions
    
sample_size = 50


Zs, Zs_mean, Zs_std = modelDrop.sample(Z_tens, N=sample_size)


data1 = torch.tensor([0, 3]).float().unsqueeze(0)
data2 = torch.tensor([0, 0]).float().unsqueeze(0)
data3 = torch.tensor([-4, 1]).float().unsqueeze(0)


samples1,_,_ = modelDrop.sample(data1)
samples2,_,_ = modelDrop.sample(data2)
samples3,_,_ = modelDrop.sample(data3)

samples1 = samples1[0].numpy()
samples2 = samples2[0].numpy()
samples3 = samples3[0].numpy()

modelDrop.eval()
with torch.no_grad():
    s1 = modelDrop(data1).item()
    s2 = modelDrop(data2).item()

    s3 = modelDrop(data3).item()

modelDrop.eval()
with torch.no_grad():
    Z_hat =  modelDrop(Z_tens)

plt.figure(figsize=(9, 8))
plt.subplot(211)
plt.contourf(xx, yy, expit(Z_hat.data.reshape(xx.shape)), alpha=0.9, cmap=cm, vmin=0, vmax=1)
plt.contour(xx, yy, expit(Z_hat.data.reshape(xx.shape)), alpha=1, cmap=cm, vmin=0, vmax=1)
plt.axis('scaled')
plt.title("Prediction using all neurons")

plt.subplot(223)
plt.contourf(xx, yy, Zs_mean.data.reshape(xx.shape), alpha=0.9, cmap=cm, vmin=0, vmax=1)
plt.colorbar()
plt.contour(xx, yy, Zs_mean.data.reshape(xx.shape), alpha=1, cmap=cm, vmin=0, vmax=1)
plt.axis('scaled')
plt.title("Average prediction using dropout\n (100 samples)")

plt.subplot(224)
plt.contourf(xx, yy, Zs_std.data.reshape(xx.shape), alpha=1)
plt.colorbar()
plt.axis('scaled')
plt.title("Standard deviation on predictions using dropout\n (100 samples)")

pts = np.asarray(plt.ginput(3, timeout=-1)) 
plt.show()

pts_tensor = torch.tensor(pts).float()

print(pts_tensor)

input_samples, input_mus, input_sds = modelDrop.sample(pts_tensor)

plt.figure()


plt.subplot(121)
plt.contourf(xx, yy, Zs_mean.data.reshape(xx.shape), alpha=0.6, cmap=cm)
plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.6, cmap=cm)
plt.colorbar()

bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )

plt.text(pts[0][0], pts[0][1], s="A", bbox=bbox)
plt.text(pts[1][0], pts[1][1], s="B", bbox=bbox)
plt.text(pts[2][0], pts[2][1], s="C", bbox=bbox)

locChr = ord("A")

for i in range(len(input_samples)):
    plt.subplot(len(input_samples),2, (i+1)*2)
    plt.title("Distribution at " + chr(locChr + i))
    plt.hist(x=input_samples[i], bins=20, range=(0,1), alpha=0.6)
    plt.axvline(input_mus[i].numpy())
plt.show()





print(pts)