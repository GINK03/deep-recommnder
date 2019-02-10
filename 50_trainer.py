import glob
import math
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.l1 = nn.Linear(
            in_features=input_size, out_features=1024, bias=True)
        self.l2 = nn.Linear(
            in_features=1024, out_features=512, bias=True)
        self.l3 = nn.Linear(
            in_features=512, out_features=512, bias=True)
        self.l4 = nn.Linear(
            in_features=512, out_features=512, bias=True)
        self.l5 = nn.Linear(
            in_features=512, out_features=1024, bias=True)
        self.l6 = nn.Linear(
            in_features=1024, out_features=input_size, bias=True)

    def encode(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x

    def decode(self, x):
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = F.dropout(x, p=0.8)
        x = self.decode(x)
        return x


def myLoss(output, target):
    # loss = nn.MSELoss()(output, target)
    loss = torch.sqrt(torch.mean((output-target)**2))
    return loss


if __name__ == '__main__':
    movie_index = json.load(open('./works/defs/smovie_index.json'))
    model = AutoEncoder(len(movie_index)).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    testData = pickle.load(open('works/dataset/test_01.pkl', 'rb')).todense()

    for epoch in range(12):
        for fnTrain in glob.glob(f'works/dataset/train_*.pkl'):
            trainData = pickle.load(open(fnTrain, 'rb'))
            height, width = trainData.shape
            for index, miniTrain in enumerate(np.array_split(trainData.todense(), height//128)):
                #print('miniTrain', fnTrain, index)
                inputs = Variable(torch.from_numpy(miniTrain)).float().cuda()
                predict = model(inputs)
                loss = myLoss(predict, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if index % 100 == 0:
                    inputs = Variable(torch.from_numpy(
                        testData[:7000])).float().cuda()
                    loss = myLoss(inputs, model(inputs))
                    print(math.sqrt(loss.data.cpu().numpy()))
                    del inputs
        torch.save(model.state_dict(), f'conv_autoencoder_{epoch:04d}.pth')
