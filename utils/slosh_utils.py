
from torch.utils import data
import torch.utils.data as utils
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class NPZ_Dataset(data.Dataset):
    def __init__(self, filenames,labels, outlier_mode='none'):

        self.filenames = filenames
        self.labels=labels
        self.indexes = np.arange(len(self.filenames))
        self.outlier_mode = outlier_mode

        assert len(self.indexes) == len(self.filenames) == len(self.labels)

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates ONE sample of data'

        batch_filenames = self.filenames[index]
        # Generate data
        X, y = self.data_generation(batch_filenames)

        return X.copy(), y#, batch_filenames
        # print('y: ', y)
        # print('y_sigma: ', y_sigma)

    def data_generation(self, batch_filenames):
        data = np.load(batch_filenames)
        im = data['im']
        try:
            y = data['label']
        except:
            y = data['det']

        return im, y

class SLOSH_Embedding(nn.Module):
    def __init__(self, embed_size=2):
        super(SLOSH_Embedding, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, padding=3)  # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, padding=2)  # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # same padding 2P = K-1

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.5)
        self.embed_size = embed_size
        self.linear1 = nn.Linear(int(16 * 16 * 16), self.embed_size)

    def forward(self, input_image):
        conv1 = self.conv1(input_image.unsqueeze(1)) # (N, C, H, W)
        conv1 = F.leaky_relu(conv1)
        conv1 = self.pool1(conv1)

        conv2 = F.leaky_relu(self.conv2(conv1))
        conv2 = self.pool2(conv2)


        conv3 = F.leaky_relu(self.conv3(conv2))
        conv3 = self.pool3(conv3)
        conv3 = self.drop1(conv3)

        linear1 = self.linear1(conv3.view(conv3.size()[0], -1))
        return linear1, conv3.view(conv3.size()[0], -1)

    def get_embedding(self, x):
        return self.forward(x)
