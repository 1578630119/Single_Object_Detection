import re
import requests
from urllib import error
import os
import cv2

num = 0
numPicture = 0
file = 'Crawling_result'
List = []
headers = {
    'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
    'Connection': 'keep-alive',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
    'Upgrade-Insecure-Requests': '1'
}
A = requests.Session()
A.headers = headers


def dowmloadPicture(html, keyword):
    global num
    # t =0
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)  #正则表达式找到目标图片的url
    for each in pic_url:
        print('下载第' + str(num + 1) + '张图片')
        try:
            if each is not None:
                pic = requests.get(each, timeout=7)
            else:
                continue
        except BaseException:
            continue
        else:
            string = file + r'/' + str(num) + '.jpg'
            print(string)
            fp = open(string, 'wb')
            fp.write(pic.content)
            fp.close()
            num += 1
        if num >= numPicture:
            return

def Crawling_images():
    word = input("请输入搜索词: ")
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='
    numPicture = int(input('请输入下载的数量 '))
    t = 0
    tmp = url
    while t < numPicture:
        try:
            url = tmp + str(t)
            # 这里搞了下
            result = A.get(url, timeout=10, allow_redirects=False)

        except error.HTTPError as e:
            print('网络错误')
            t = t + 1
        else:
            dowmloadPicture(result.text, word)
            t = t + 1

def Arrange_image():
    i=0
    for img_name in os.listdir("Crawling_result"):
        img=cv2.imread("Crawling_result/"+img_name)
        if img is not None:
            i_name=str(i).zfill(6)
            cv2.imwrite('Background/'+i_name+'.jpg',img)
            i=i+1
        os.remove("Crawling_result/"+img_name)


if __name__=="__main__":
    Crawling_images() #在网上爬取图片
    Arrange_image() #整理下载好的图像，只保留可以加载的图片