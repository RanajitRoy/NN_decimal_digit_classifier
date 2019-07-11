import os
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.io import imread


def image_to_np(sz):
    img_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + os.sep + 'img' + os.sep
    list_img = os.listdir(img_path)
    arr = 255 - imread(img_path + list_img[0], True) * 255
    arr = resize(arr, (sz, sz), anti_aliasing=True)
    plt.matshow(arr)
    plt.show()
    return arr
