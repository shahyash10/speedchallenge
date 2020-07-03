import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from NeuralNet.model import Model
from torchvision import transforms, utils
from NeuralNet.dataset_class import DatasetClass
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])])

data = DatasetClass("data/train_images/", "data/train.txt", transform)
batch_size = 4
num_workers = 0
valid_size = 0.6
# obtain training indices that will be used for validation
num_train = len(data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=num_workers)


criterion = nn.MSELoss()
Network = Model()
# specify optimizer
optimizer = optim.Adam(Network.parameters(), lr=3e-3, weight_decay=3e-5)
# training loop
# number of epochs to train the model
n_epochs = 30
print("\n---- Started training the model ------\n")
for epoch in range(1, n_epochs+1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    print("------ Epoch ", epoch, " ---------")

    ###################
    # train the Network #
    ###################
    Network.train()
    for image, label in train_loader:
        image = image.float()
        label = label.float()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the Network
        prediction = Network(image)
        # calculate the batch loss
        loss = criterion(prediction, label)
        # backward pass: compute gradient of the loss with respect to Network parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()
    ######################
    # validate the Network #
    ######################
    Network.eval()
    for image, label in valid_loader:
        image = image.float()
        label = label.float()
        # forward pass: compute predicted outputs by passing inputs to the Network
        prediction = Network(image)
        # calculate the batch loss
        loss = criterion(prediction, label)
        # update average validation loss
        valid_loss += loss.item()

    # calculate average losses
    train_loss = train_loss/len(train_loader)
    valid_loss = valid_loss/len(valid_loader)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n'.format(
        epoch, train_loss, valid_loss))
print("\n-----Finished training the model------")


# save Network
torch.save(Network.state_dict(), 'Model.pt')
