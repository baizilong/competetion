import logging
import numpy as np
import random
import datahelper
import torch
import model
import trainer
import torch.utils.data as Data
from sklearn.model_selection import StratifiedKFold,KFold
from torch.backends import cudnn
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True

logging.basicConfig(level=logging.INFO,
                    # filename='./log/log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
datafile='./train.mymemmap'#train.csv文件位置
num_labels=3
batchsize=128
output_dir='./baseline/test'#模型保存目录
UsingGPU=True
randomseed=5
seqlenth=2600
epoch=10
lr=3e-3
k_fold=10#折数
if __name__ == '__main__':
    random.seed(randomseed)
    np.random.seed(randomseed)
    torch.manual_seed(randomseed)
    if UsingGPU:
        torch.cuda.manual_seed_all(randomseed)
    #train_rows,labels =datahelper.getdata(datafile,range(0,56),True)
    train_rows,labels =datahelper.gettraindata(datafile)
    stratified_folder = KFold(n_splits=k_fold, random_state=randomseed, shuffle=True)
    for k, (train_index, eval_index) in enumerate(stratified_folder.split(train_rows)):
        train_temp=[]
        label_temp=[]
        for i in train_index:
            train_temp.append(train_rows[i])
            label_temp.append(labels[i])
        train_data = torch.from_numpy(np.array(train_temp,dtype=np.float32))
        train_labels = torch.from_numpy(np.array(label_temp))
        train_dataset = Data.TensorDataset(train_data,train_labels)
        train_loader = Data.DataLoader(
            # 从数据库中每次抽出batch size个样本
            dataset=train_dataset,
            batch_size=batchsize,
            shuffle=False,
            num_workers=2,
        )
        eval_data = torch.from_numpy(np.array([train_rows[i] for i in eval_index],dtype=np.float32))
        eval_labels = torch.from_numpy(np.array([labels[i] for i in eval_index]))
        eval_dataset = Data.TensorDataset(eval_data, eval_labels)
        eval_loader = Data.DataLoader(
            # 从数据库中每次抽出batch size个样本
            dataset=eval_dataset,
            batch_size=batchsize,
            shuffle=False,
            num_workers=2,
        )

        logger.info("Starting{}-fold".format(k))
        num_train_steps = int(len(train_index) * epoch / batchsize)
        mymodel =model.model(seqlenth,featuresize=4,seqembedding=3,dropout=0.5)
        optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)
        trainer.train(mymodel, optimizer, output_dir+'_'+str(k), epoch, train_loader, eval_loader, UsingGPU=UsingGPU,
                      min_f1score=0.98, maxtokeep=3, CVAfterEpoch=1,classnum=num_labels)



