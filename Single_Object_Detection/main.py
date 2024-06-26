import os
import numpy as np
from Dataloader import ImgDataset
import os
import numpy as np
from Dataloader import ImgDataset
from model import Net
from torch import nn,utils,optim
import torch
from tqdm import tqdm
import cv2


train_batch_size=8  #训练集、验证集、测试集批大小
eval_batch_size=32
test_batch_size=32

with open('data/train.txt') as f:
    train_lines=f.readlines()
with open('data/test.txt') as f:
    test_lines=f.readlines()
eval_lines=train_lines[:int(len(train_lines)*0.2)]  #将原本的训练集划分为训练集和验证集，训练集：验证集=4：1
train_lines=train_lines
train_dataset=ImgDataset(train_lines)
eval_dataset=ImgDataset(eval_lines)
test_dataset=ImgDataset(test_lines)

# 创建数据加载器
train_loader = utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
eval_loader = utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
test_loader = utils.data.DataLoader(test_dataset, batch_size=test_batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Net()
model=model.to(device)
criterion=nn.L1Loss()
criterion=criterion.to(device)
optimizer=optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3)


def train(epoch, epochs):
    # 训练模型
    train_loss = 0
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', mininterval=0.3)
    for batch_idx, (data, target) in enumerate(train_loader):  # 批次，输入数据，标签
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(**{'train loss': train_loss / (batch_idx + 1)})
        pbar.update(1)
    return train_loss / (batch_idx + 1)

def eval(epoch,epochs):
    #测试模型
    model.eval()
    pbar = tqdm(total=len(eval_loader), desc=f'Epoch {epoch + 1}/{epochs}', mininterval=0.3)
    eval_loss = 0
    with torch.no_grad(): #仅测试模型，禁用梯度计算
        for batch_idx, (data, target) in enumerate(eval_loader):
            data=data.to(device)
            target=target.to(device)
            output = model(data)
            eval_loss += criterion(output, target).item()
            pbar.set_postfix(**{'eval loss': eval_loss / (batch_idx + 1)})
            pbar.update(1)
    return eval_loss/(batch_idx + 1)


def model_fit(epochs):
    best_loss = 1e7
    for epoch in range(epochs):
        train_loss = train(epoch, epochs)
        eval_loss = eval(epoch, epochs)
        print('\nEpoch: {}\tTrain Loss: {:.6f}\tEval Loss: {:.6f}'.format(epoch + 1, train_loss, eval_loss))
        if eval_loss < best_loss:
            best_loss = eval_loss
        torch.save(model.state_dict(), 'model.pth')#保存在验证集上预测效果最好的模型权重
def test():
    #如果已经训练好了权重，模型直接加载权重文件进行测试#
    model_test=Net()
    model_test.load_state_dict(torch.load('model.pth',map_location=device))
    model_test.eval()
    model_test=model_test.to(device)
    test_loss = 0
    with torch.no_grad():  # 仅测试模型，禁用梯度计算
        for batch_idx, (data, target) in enumerate(eval_loader):
            data = data.to(device)
            target = target.to(device)
            output = model_test(data)
            test_loss += criterion(output, target).item()
    print('Test Loss:',test_loss/(batch_idx+1))

def demo():
    #test()
    model_test=Net()
    model_test.load_state_dict(torch.load('model.pth'))
    model_test.eval()
    with torch.no_grad():
        for line in test_lines:
            line=line.split(' ')
            image=cv2.imread('data/'+line[0])
            data= np.array([np.transpose(np.array(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), dtype=np.float32)/255.0, (2, 0, 1))])
            output = model_test(torch.Tensor(data))
            box = np.array(list(map(float, line[1].split(','))))
            box = torch.Tensor(np.array([box[:4]]))
            loss=criterion(output,box)
            output = np.uint32(output.detach().numpy())
            output[...,[0,2]],=np.clip(output[...,[0,2]],0,416)
            output[..., [1, 3]], = np.clip(output[..., [1, 3]], 0, 320)
            image=cv2.rectangle(image,output[0,:2],output[0,2:],(0,0,255),2)
            cv2.imshow('a',image)
            cv2.waitKey(2000)



if __name__=="__main__":
    model_fit(200)
    test()
    demo()










