import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

learning_rate = 0.001  # 学习率
momentum = 0.5  # 使用optim.SGD（随机梯度下降）优化器时，momentum是一个重要的参数。它代表了动量（Momentum）的大小，是动量优化算法中的一个关键概
train_batch_size = 32
eval_batch_size = 128
test_batch_size = 128
trainset = torchvision.datasets.CIFAR10('./data/', train=True, download=True,  # 训练集下载
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),  # 转换数据类型为Tensor
                                        ]))

# ------------------------------------------------------------------#
#       将训练集再划分为训练集和测试集（训练集：测试集=4：1）    #
# ------------------------------------------------------------------#
train_size = len(trainset)
indices = list(range(train_size))

# 划分索引
split = int(0.8 * train_size)
train_indices, val_indices = indices[:split], indices[split:]

# 创建训练集和验证集的子集
trainset_subset = torch.utils.data.Subset(trainset, train_indices)
valset_subset = torch.utils.data.Subset(trainset, val_indices)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(trainset_subset, batch_size=train_batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(valset_subset, batch_size=eval_batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data/', train=False, download=True,  # 测试集下载
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                 ])),
    batch_size=test_batch_size, shuffle=True)


def Net():
    resnet18 = torchvision.models.resnet18(pretrained=True) #调用ResNet18,并加载预训练权重
    #print(resnet18)    #打印模型结构，可以看见最后一层输出的1000个结果
    resnet18.fc = nn.Linear(512, 10)  # 重搭全连接层
    return resnet18


model = Net()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # 优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测电脑是否能使用cuda训练，不行则使用cpu
model = model.to(device)
Train_Loss = []
Eval_Loss = []


def train(epoch, epochs):
    # 训练模型
    train_loss = 0
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', mininterval=0.3)
    for batch_idx, (data, target) in enumerate(train_loader):  # 批次，输入数据，标签
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()  # 清空优化器中的梯度
        output = F.log_softmax(model(data), dim=1)  # 前向传播，获得当前模型的预测值
        loss = F.nll_loss(output, target)  # 真实值和预测值之前的损失
        loss.backward()  # 反向传播，计算损失函数关于模型中参数梯度
        optimizer.step()  # 更新模型中参数
        # 输出当前训练轮次，批次，损失等
        Train_Loss.append(loss.item())
        train_loss += loss.item()
        pbar.set_postfix(**{'train loss': train_loss / (batch_idx + 1)})
        pbar.update(1)
    return train_loss / (batch_idx + 1)


def eval(epoch, epochs):
    # 测试模型
    model.eval()
    pbar = tqdm(total=len(eval_loader), desc=f'Epoch {epoch + 1}/{epochs}', mininterval=0.3)
    eval_loss = 0
    with (torch.no_grad()):  # 仅测试模型，禁用梯度计算
        for batch_idx, (data, target) in enumerate(eval_loader):
            data = data.to(device)
            target = target.to(device)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target).item()
            eval_loss += loss

            Eval_Loss.append(loss)
            pbar.set_postfix(**{'eval loss': eval_loss / (batch_idx + 1)})
            pbar.update(1)
    return eval_loss / (batch_idx + 1)


def model_fit(epochs):
    best_loss = 1e7
    for epoch in range(epochs):
        train_loss = train(epoch, epochs)
        eval_loss = eval(epoch, epochs)
        print('\nEpoch: {}\tTrain Loss: {:.6f}\tEval Loss: {:.6f}'.format(epoch + 1, train_loss, eval_loss))
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), 'model.pth')
    with open("Train_Loss.txt", 'w') as f:
        for num in Train_Loss:
            f.write(str(num) + '\n')
    with open("Eval_Loss.txt", 'w') as f:
        for num in Eval_Loss:
            f.write(str(num) + '\n')


def test():
    # 如果已经训练好了权重，模型直接加载权重文件进行测试#
    model_test = Net()
    model_test.load_state_dict(torch.load('model.pth'))
    model_test.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = F.log_softmax(model_test(data),dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def demo():
    with open('Train_Loss.txt') as f:
        lines = f.readlines()
    lines = np.array(lines, dtype=np.float32)
    iters = range(1, len(lines) + 1)  # 训练步数（或迭代次数），这里简单用1到损失值的数量来表示

    # 使用plot函数绘制损失图
    plt.plot(iters, lines, marker='.')  # marker='o' 表示在数据点上显示圆圈

    # 添加标题和坐标轴标签
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # 显示网格（可选）
    plt.grid(True)

    # 显示图形
    plt.show()
    model_test = Net()
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = model_test(example_data)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":
    #model_fit(100)  #训练100轮
    test()          #测试模型
    demo()          #输出损失图，测试样本
