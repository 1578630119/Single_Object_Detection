import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image
def preprocess_input(image):
    image /= 255.0      #数据归一化
    return image

class ImgDataset(Dataset):
    def __init__(self, annotation_lines, input_shape=[320, 416]):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape      #设定图片的尺寸

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image, y = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2])
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = y[:4]
        label = y[-1]
        return image, box

    
    def get_random_data(self, annotation_line, input_shape):
        line = annotation_line.split()

        image = Image.open('data/'+line[0])
        image = cvtColor(image)     #确保图片是三通道，如果不是则转换成三通道格式

        iw, ih = image.size #图片原始尺寸
        h, w = input_shape  #设定图片的高宽

        box = np.array(list(map(int, line[1].split(','))))
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2  #计算图片尺寸调整比例
        # ---------------------------------#
        #   将图像多余的部分加上灰条
        # ---------------------------------#
        image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)

        # ---------------------------------#
        #   因对原图像进行尺寸调整，还需对真实框进行调整
        # ---------------------------------#
        box[0] = box[0] * nw / iw + dx
        box[2] = box[2] * nw / iw + dx
        box[1] = box[1] * nh / ih + dy
        box[3] = box[3] * nh / ih + dy
        box[0:2][box[0:2] < 0] = 0
        box[2:3][box[2:3] > w] = w
        box[3:4][box[3:4] > h] = h

        return image_data, box


