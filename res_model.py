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
class resnet(nn.Module):
    def __init__(self,chinel,kernel_shape,stride,padding):
        super(resnet, self).__init__()
        self.layer1 = nn.Conv1d(chinel, chinel, kernel_shape,stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(chinel)
        self.layer2 = nn.Conv1d(chinel, chinel, kernel_shape, stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(chinel)
    def forward(self, x, labels=None):
        outputs = F.relu(self.bn1(self.layer1(x)))
        outputs = self.layer1(outputs)
        outputs = F.relu(self.bn2(outputs)+x)
        return outputs

class model(nn.Module):
    def __init__(self, seqlenth, featuresize=9, seqembedding=3, dropout=0.2):
        super(model, self).__init__()
        self.lossfun = nn.CrossEntropyLoss()
        self.seqembedding = nn.Parameter(data=torch.rand([seqembedding, seqlenth]), requires_grad=True)
        self.bn0 = nn.BatchNorm1d(featuresize + seqembedding)
        self.layer1 = nn.Conv1d(featuresize + seqembedding, 256, 3, 1, padding=0)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2=resnet(256,3,1,1)
        self.layer3 = resnet(256, 3, 1, 1)
        self.layer4 = resnet(256, 3, 1, 1)
        self.layer5 = resnet(256, 3, 1, 1)
        self.layer6 = resnet(256, 3, 1, 1)
        self.layer7 = resnet(256, 3, 1, 1)
        self.layer8 = resnet(256, 3, 1, 1)
        self.layer9 = resnet(256, 3, 1, 1)


        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2560, 3)

    def forward(self, x, labels=None):
        seqembedding = self.seqembedding.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([x, seqembedding], 1)

        x=F.relu(self.bn1(self.layer1(x)))
        x=self.layer2(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.layer3(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.layer4(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.layer5(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.layer6(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.layer7(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.layer8(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.layer9(x)
        x = F.max_pool1d(x, 2, 2)

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
