import os
import numpy as np
from Dataloader import ImgDataset
from model import Net_res,DIou_Loss
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
criterion=DIou_Loss()
criterion=criterion.to(device)



def train(epoch, epochs,model,optimizer):
    # 训练模型
    train_loss = 0
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs//2}', mininterval=0.3)
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

def eval(epoch,epochs,model,optimizer):
    #测试模型
    model.eval()
    pbar = tqdm(total=len(eval_loader), desc=f'Epoch {epoch + 1}/{epochs//2}', mininterval=0.3)
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
    model = Net_res()
    model = model.to(device)
    for param in model.features.parameters():   #冻结网络的features层，即resnet的第一层到layer2层
        param.requires_grad = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs//2):
        train_loss = train(epoch, epochs,model,optimizer)
        eval_loss = eval(epoch, epochs,model,optimizer)
        print('\nEpoch: {}\tTrain Loss: {:.6f}\tEval Loss: {:.6f}'.format(epoch + 1, train_loss, eval_loss))
        if eval_loss < best_loss:
            best_loss = eval_loss
        torch.save(model.state_dict(), 'model.pth')#保存在验证集上预测效果最好的模型权重


    for param in model.features.parameters():   #解冻features层
        param.requires_grad = True
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epochs // 2,epochs):
        train_loss = train(epoch, epochs, model, optimizer)
        eval_loss = eval(epoch, epochs, model, optimizer)
        print('\nEpoch: {}\tTrain Loss: {:.6f}\tEval Loss: {:.6f}'.format(epoch + 1, train_loss, eval_loss))
        if eval_loss < best_loss:
            best_loss = eval_loss
        torch.save(model.state_dict(), 'model.pth')  # 保存在验证集上预测效果最好的模型权重

def iou(b1, b2):
    b1_wh = b1[:, 2:4] - b1[:, :2]
    b2_wh = b2[:, 2:4] - b2[:, :2]
    inter_x1 = torch.max(b1[:, 0], b2[:, 0])
    inter_y1 = torch.max(b1[:, 1], b2[:, 1])
    inter_x2 = torch.min(b1[:, 2], b2[:, 2])
    inter_y2 = torch.min(b1[:, 3], b2[:, 3])

    # ----------------------------------------------------#
    #   求真实框和预测框所有的iou
    # ----------------------------------------------------#
    intersect_area = (torch.clamp(inter_x2 - inter_x1, min=0) + 1) * (torch.clamp(inter_y2 - inter_y1, min=0) + 1)
    b1_area = (b1_wh[:, 0] + 1) * (b1_wh[:, 1] + 1)
    b2_area = (b2_wh[:, 0] + 1) * (b2_wh[:, 1] + 1)
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / union_area
    return iou
def test():
    #如果已经训练好了权重，模型直接加载权重文件进行测试#
    model_test=Net_res()
    model_test.load_state_dict(torch.load('model.pth',map_location=device))
    model_test.eval()
    model_test=model_test.to(device)
    test_loss = 0
    correct=0
    with torch.no_grad():  # 仅测试模型，禁用梯度计算
        for batch_idx, (data, target) in enumerate(eval_loader):
            data = data.to(device)
            target = target.to(device)
            output = model_test(data)
            test_loss += criterion(output, target).item()
            result = iou(output, target)
            result = torch.where(result > 0.3, 1, 0)
            correct = correct + result.sum()
    print('Accuracy:{:.2f}%'.format((correct / ((batch_idx+1)*test_batch_size)).to('cpu').detach().numpy()*100))
    print('Test Loss:',test_loss/(batch_idx+1))

def demo():
    #test()
    model_test=Net_res()
    model_test.load_state_dict(torch.load('model.pth',map_location=device))
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
            #cv2.waitKey(2000)
            cv2.imwrite('Crawling_result/'+line[0],image)



if __name__=="__main__":
    #model_fit(200)
    #test()
    demo()
