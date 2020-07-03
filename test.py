import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from NeuralNet.model import Model
from torchvision import transforms

filename = "data/test.txt"
if not os.path.isfile(filename):
    open(filename, 'w').close()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])])

Network = Model()
Network.load_state_dict(torch.load(
    'Model.pt', map_location=torch.device('cpu')))
# eval mode
Network.eval()

test_data = os.listdir("data/test_images/")
test_results = np.ones((len(test_data)+1, 1))

for i in range(len(test_data)):
    img_adr = "data/test_images/" + str(i) + ".png"
    img = cv2.imread(img_adr)
    img = transform(img)
    img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    ans = Network(img)
    test_results[i+1, 0] = ans.item()

# 0th frame speed will be hardcoded and set equal to the first frame speed
test_results[0, 0] = test_results[1, 0]

a_file = open("data/test.txt", "w")
for row in test_results:
    np.savetxt(a_file, row)

a_file.close()
