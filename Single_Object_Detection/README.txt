Data_Download.py 是下载背景图片，下载的图片类型和数量由自己设定
generate_dataset.py 是生成图片，将小黄人粘贴在背景图片上，同时会生成标签文件，图片和标签train.txt test.txt文件都存放在data文件夹中
Dataloader.py是数据加载代码
model.py是模型代码
main.py  里面是训练和测试模型代码
CIFAR10_mask.py 是ResNet18实现CIFAR10分类任务
main_resnet.py 是利用resnet18重新搭建神经模型训练小黄人目标检测任务，同时使用冻结部分参数训练模型的方式
