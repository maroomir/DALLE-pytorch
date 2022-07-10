import math
import os
import os.path

import numpy
import torchvision.transforms
import matplotlib.pyplot

from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda, Resize, CenterCrop, ToTensor
from PIL import Image


def load_dataset(img_path: str, img_size: int = 128) -> ImageFolder:
    transform = torchvision.transforms.Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize(img_size),
        CenterCrop(img_size),
        ToTensor()
    ])
    return torchvision.datasets.ImageFolder(img_path, transform=transform)


def draw_images(images: list):
    rows = int(math.sqrt(len(images)))
    cols = int(len(images) / rows)
    totals = []
    for j in range(rows):
        row = []
        for i in range(cols):
            row += [images[j * cols + i]]
        totals.append(numpy.concatenate(row, axis=1))
    totals = numpy.concatenate(totals)
    matplotlib.pyplot.figure(figsize=(10, 10))
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.imshow(totals)
    matplotlib.pyplot.show()


def print_labels(labels: list, sep='_'):
    rows = int(math.sqrt(len(labels)))
    cols = int(len(labels) / rows)
    for j in range(rows):
        row = []
        for i in range(cols):
            row += [sep.join(labels[j * cols + i])]
        print(':'.join(row))


IMG_EXTS = ['.png', '.jpg', '.bmp']


def load_labels(source: (str, ImageFolder), sep='_', trace=False):
    res = []
    split, splitext = os.path.split, os.path.splitext
    if isinstance(source, ImageFolder):
        files = [split(splitext(pth)[0])[-1] for pth, _ in source.samples]
        res = [f.split(sep) for f in files]
    elif isinstance(source, str) and os.path.exists(source):
        for roots, dirs, files in os.walk(source):
            if len(files) > 0:
                res += [splitext(f)[0].split(sep) for f in files if splitext(f)[1] in IMG_EXTS]
    else:
        raise ValueError(f"Unreadable Source = {source}")
    if trace:
        print_labels(res)
    return res


def load_images(source: (str, ImageFolder), trace=False):
    res = []
    if isinstance(source, ImageFolder):
        res = [numpy.array(Image.open(im_path))[:, :, :3] / 255 for im_path, _ in source.imgs]
    elif isinstance(source, str) and os.path.exists(source):
        for roots, dirs, files in os.walk(source):
            if len(files) > 0:
                res += [numpy.array(Image.open(os.path.join(roots, f)))[:, :, :3] / 255
                        for f in files if os.path.splitext(f)[1] in IMG_EXTS]
    else:
        raise ValueError(f"Unreadable Source = {source}")
    if trace:
        draw_images(res)
    return res


if __name__ == "__main__":
    test_path = './data/test'
    test_set = load_dataset(img_path=test_path, img_size=128)
    test_images = load_images(source=test_set, trace=True)
    test_labels = load_labels(source=test_set, trace=True)
