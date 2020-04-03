import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((channel,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )


    def forward(self, x):
        #b, c, k = x.size()
        y=torch.squeeze(self.avg_pool(x),-1)
        #y=torch.squeeze(F.max_pool1d(x,k,1),-1)
        y = self.fc(y)#.view(b, c, 1, 1)
        return y

def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU6(inplace=True),
    )

def ConvBN(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm1d(out_channels))

class InceptionV2ModuleA(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4reduce,out_channels4,out_channels5):
        super(InceptionV2ModuleA, self).__init__()

        self.branch1 = ConvBN(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBN(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=5, padding=2),
            ConvBN(in_channels=out_channels3, out_channels=out_channels3, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels4reduce,kernel_size=1),
            ConvBNReLU(in_channels=out_channels4reduce, out_channels=out_channels4, kernel_size=7, padding=3),
            ConvBN(in_channels=out_channels4, out_channels=out_channels4, kernel_size=7, padding=3),
        )

        self.branch5 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            ConvBN(in_channels=in_channels, out_channels=out_channels5, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out5 = self.branch5(x)
        out = F.relu6(torch.cat([out1, out2, out3, out4, out5], dim=1)+x)
        return out

class model(nn.Module):
    def __init__(self, seqlenth, featuresize=9, seqembedding=3, dropout=0.2):
        super(model, self).__init__()
        self.lossfun = nn.CrossEntropyLoss()
        self.seqembedding = nn.Parameter(data=torch.rand([seqembedding, seqlenth]), requires_grad=True)

        self.bn0 = nn.BatchNorm1d(featuresize + seqembedding)
        self.layer0=ConvBN(featuresize + seqembedding,256,3,1,1)

        self.block0=InceptionV2ModuleA(256,32,64,64,64,64,64,64,32)
        self.block1 = InceptionV2ModuleA(256, 32, 64, 64, 64, 64, 64, 64, 32)

        self.layer1 = ConvBNReLU(256, 256, 3, 1,0)

        self.layer2 = ConvBNReLU(256, 256, 5, 1,0)

        self.layer3 = ConvBNReLU(256, 256, 7, 1,0)

        self.layer4 = ConvBNReLU(256, 256, 9, 1,0)


        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(1280, 3)

    def forward(self, x, labels=None):
        seqembedding = self.seqembedding.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([x, seqembedding], 1)
        x = self.bn0(x)
        x=self.layer0(x)
        x=self.block0(x)
        x = self.block0(x)

        x=self.layer1(x)
        x = F.max_pool1d(x, 2, 2)

        x = self.layer2(x)
        x = F.max_pool1d(x, 4, 4)

        x = self.layer4(x)
        x = F.max_pool1d(x, 6, 6)

        x = self.layer4(x)
        x = F.max_pool1d(x, 8, 8)

        x = torch.reshape(x, [x.shape[0], -1])
        x = self.dropout(x)

        logist = self.fc(x)

        if labels is not None:
            loss = self.lossfun(logist, torch.squeeze(labels, 1))
            return logist, loss
        return logist

    def inference(self, x):
        logist = self.forward(x)
        return list(torch.argmax(logist, dim=1).cpu().numpy()), list(
            torch.softmax(logist, dim=1).cpu().detach().numpy())


def save(model, step, outputdir, MaxModelCount=5):
    checkpoint = []
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    if os.path.exists(os.path.join(outputdir, 'checkpoint')):
        with open(os.path.join(outputdir, 'checkpoint')) as f:
            checkpoint = f.readlines()
    checkpoint.append('model_' + str(step) + '.kpl\n')
    logging.info('Saving model as \"' + checkpoint[-1].strip('\n') + '\"')
    while len(checkpoint) > MaxModelCount:
        os.remove(os.path.join(outputdir, checkpoint[0].strip('\n')))
        checkpoint.pop(0)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
    }, os.path.join(outputdir, checkpoint[-1].strip('\n')))
    with open(os.path.join(outputdir, 'checkpoint'), 'w') as f:
        f.writelines(checkpoint)


def load(model, outputdir):
    checkpoint = []
    if os.path.exists(os.path.join(outputdir, 'checkpoint')):
        with open(os.path.join(outputdir, 'checkpoint')) as f:
            checkpoint = f.readlines()
    if len(checkpoint) < 1:
        return model, 1, 0
    modelpath = os.path.join(outputdir, checkpoint[-1].strip('\n'))
    logging.info('Restoring model from \"' + checkpoint[-1].strip('\n') + '\"')
    dic = torch.load(modelpath, map_location='cpu')
    model.load_state_dict(dic['model_state_dict'])
    step = dic['step']
    return model, step
