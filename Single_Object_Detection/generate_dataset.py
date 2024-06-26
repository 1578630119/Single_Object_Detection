import os
import random

from PIL import Image
import numpy as np

bg_path = "Background"
bg_names = os.listdir(bg_path)
bg_w=416
bg_h=320


def Dataset_generate():
    #背景图片中前80%和小黄人图片中前80%用来生成训练集和验证集，后百分之20用来生成测试集
    nums = len(bg_names)
    x = 0
    ftrain = open('data/train.txt', 'w')
    ftest = open('data/test.txt', 'w')
    for i in range(nums):
        background = Image.open("{0}/{1}".format(bg_path, bg_names[i]))
        shape = np.shape(background)    #检测背景通道数
        if len(shape) == 3:
            background = background
        else:
            background=background.convert('RGB')
        background_resize = background.resize((bg_w,bg_h))   #统一背景图片大小
        for k in range(random.randint(5, 9)):   #针对每一张背景图片，随机和小黄人图片生成n张图片
            background_new=background_resize.copy()
            name = np.random.randint(1, 21) #随机选择小黄人图片
            img_font = Image.open("yellow/{0}.png".format(name))
            ran_w = np.random.randint(60, 150)
            ran_h = np.random.randint(80, 200)
            img_new = img_font.resize((ran_w, ran_h))

            ran_x1 = np.random.randint(0, bg_w - ran_w)      #小黄人随机粘贴在背景图片上的坐标
            ran_y1 = np.random.randint(0, bg_h - ran_h)

            r, g, b, a = img_new.split()
            background_new.paste(img_new, (ran_x1, ran_y1), mask=a)  #小黄人粘贴在背景图片上

            ran_x2 = ran_x1 + ran_w
            ran_y2 = ran_y1 + ran_h #小黄人右下角坐标

            background_new.save("data/{0}.jpg".format(str(x).zfill(6)))  #保存图片
            if i < int(nums * 0.8):
                ftrain.write("{}.jpg".format(str(x).zfill(6)) + " " + str(ran_x1) + "," + str(ran_y1) + "," + str(
                    ran_x2) + "," + str(ran_y2) + "," + str(name-1) + "\n")
            else:
                ftest.write("{}.jpg".format(str(x).zfill(6)) + " " + str(ran_x1) + "," + str(ran_y1) + "," + str(
                    ran_x2) + "," + str(ran_y2) + "," + str(name-1) + "\n")   #保存标签
            x += 1
    ftrain.close()
    ftest.close()

if __name__ == "__main__":
    Dataset_generate() #生成图片