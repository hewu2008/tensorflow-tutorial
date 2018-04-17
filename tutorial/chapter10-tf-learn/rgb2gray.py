#encoding:utf8

from PIL import Image

if __name__ == "__main__":
    im = Image.open('E:/Project/CPP/longan-detect/data/images/CAMERA0.bmp');
    im1 = im.convert('L')
    im1.save("E:/Project/CPP/longan-detect/data/images/CAMERA1.bmp")