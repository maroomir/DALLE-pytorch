import math
import os.path
import shutil
from abc import ABCMeta, abstractmethod

import numpy
import cairo
import matplotlib.pyplot
import matplotlib.colors
from numpy import ndarray
from tqdm import tqdm
from PIL import Image


class _CairoShape(metaclass=ABCMeta):
    def __init__(self,
                 image: ndarray,
                 size: int,
                 scale: (float, str),
                 color: (tuple, str)):
        surface = cairo.ImageSurface.create_for_data(image, cairo.FORMAT_ARGB32, size, size)
        self.context = cairo.Context(surface)
        self.context.set_antialias(cairo.ANTIALIAS_NONE)
        self.context.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        # Make the default canvas
        self.context.rectangle(0, 0, size, size)
        self.context.set_source_rgb(1, 1, 1)
        self.context.fill()
        # Init the pen tool
        if isinstance(scale, str):
            attr = {'big': 1, 'bigger': 0.8, 'smaller': 0.6, 'small': 0.4}
            scale = attr[scale]
        pen = 1 / (size * scale / 2)
        self.context.set_line_width(pen)
        # Normalization to translate scale (-1.0 ~ 1.0)
        self.context.translate(size // 2, size // 2)
        self.context.scale(scale * size / 2, scale * size / 2)
        if isinstance(color, str):
            try:
                color = reversed(matplotlib.colors.to_rgb(color))
            except:
                color = (0, 0, 0)
        self.context.set_source_rgb(*color)

    @staticmethod
    @abstractmethod
    def tag() -> str:
        pass

    @abstractmethod
    def __call__(self):
        pass

    def fill(self):
        self.context.stroke_preserve()
        self.context.fill()

    def stroke(self):
        self.context.stroke()


class Circle(_CairoShape):
    @staticmethod
    def tag():
        return 'circle'

    def __call__(self):
        self.context.arc(0, 0, 1, 0, 2 * math.pi)


class Triangle(_CairoShape):
    @staticmethod
    def tag():
        return 'triangle'

    def __call__(self):
        self.context.move_to(0, -1)
        self.context.line_to(math.sqrt(3) / 2, 0.5)
        self.context.line_to(-math.sqrt(3) / 2, 0.5)
        self.context.line_to(0, -1)


class Square(_CairoShape):
    @staticmethod
    def tag():
        return 'square'

    def __call__(self):
        self.context.rectangle(-0.9, -0.9, 1.8, 1.8)


class Rectangle(_CairoShape):
    @staticmethod
    def tag():
        return 'rectangle'

    def __call__(self):
        self.context.rectangle(-0.9, -0.5, 1.8, 1)


class Rhombus(_CairoShape):
    @staticmethod
    def tag():
        return 'rhombus'

    def __call__(self):
        self.context.move_to(0, -1)
        self.context.line_to(0.5, 0)
        self.context.line_to(0, 1)
        self.context.line_to(-0.5, 0)
        self.context.line_to(0, -1)


class Star(_CairoShape):
    @staticmethod
    def tag():
        return 'star'

    def __call__(self):
        self.context.move_to(0, -1)
        self.context.line_to(math.sqrt(3) / 2, 0.5)
        self.context.line_to(-math.sqrt(3) / 2, 0.5)
        self.context.line_to(0, -1)
        self.context.move_to(0, 1)
        self.context.line_to(math.sqrt(3) / 2, -0.5)
        self.context.line_to(-math.sqrt(3) / 2, -0.5)
        self.context.line_to(0, 1)


class Hexagon(_CairoShape):
    @staticmethod
    def tag():
        return 'hexagon'

    def __call__(self):
        self.context.move_to(0, -1)
        self.context.line_to(math.sqrt(3) / 2, -0.5)
        self.context.line_to(math.sqrt(3) / 2, 0.5)
        self.context.line_to(0, 1)
        self.context.line_to(-math.sqrt(3) / 2, 0.5)
        self.context.line_to(-math.sqrt(3) / 2, -0.5)
        self.context.line_to(0, -1)


class Crescent(_CairoShape):
    @staticmethod
    def tag():
        return 'crescent'

    def __call__(self):
        self.context.arc(0, 0, 1, -math.pi * 0.5, math.pi * 0.5)
        self.context.move_to(0, -1)
        self.context.arc(-math.sqrt(3), 0, 2, -math.pi / 6, math.pi / 6)


class _ImageTransform(metaclass=ABCMeta):
    @classmethod
    def dither(cls, mask: ndarray):
        for y in range(0, mask.shape[0] - 1):
            for x in range(1, mask.shape[1] - 1):
                pixel = mask[y][x]
                noise = round(pixel)
                error = pixel - noise
                mask[y][x] = noise
                mask[y, x + 1] += error * 112 / 256
                mask[y + 1, x - 1] += error * 48 / 256
                mask[y + 1, x] += error * 90 / 256
                mask[y + 1, x + 1] += error * 16 / 256

    def rotate(self, count: int):
        for _ in range(count):
            self.image = numpy.rot90(self.image)
        return self.image

    def __init__(self, image: ndarray, size: int):
        self.image = image
        self.size = size

    @staticmethod
    @abstractmethod
    def tag() -> str:
        pass

    @abstractmethod
    def __call__(self) -> ndarray:
        pass


class Fix(_ImageTransform):
    @staticmethod
    def tag():
        return ''

    def __call__(self):
        return self.image


class DitherHalftone(_ImageTransform):
    @staticmethod
    def tag():
        return 'halftone'

    def __call__(self):
        mask = (self.image.astype(int).sum(axis=2) != 255 * 3).astype(float)
        mask *= 0.5
        self.dither(mask)
        self.image[mask > 0.5, :] = 255
        return self.image


class DitherShaded(_ImageTransform):
    @staticmethod
    def tag():
        return 'shaded'

    def __call__(self):
        mask = (self.image.astype(int).sum(axis=2) != 255 * 3).astype(float)
        mask *= 0.3
        self.dither(mask)
        self.image[mask > 0.5, :] = 255
        return self.image


class RotateClockwise(_ImageTransform):
    @staticmethod
    def tag():
        return 'clockwise'

    def __call__(self):
        return self.rotate(count=1)


class RotateReverse(_ImageTransform):
    @staticmethod
    def tag():
        return 'reverse'

    def __call__(self):
        return self.rotate(count=2)


class RotateCounterClockwise(_ImageTransform):
    @staticmethod
    def tag():
        return 'counterclockwise'

    def __call__(self):
        return self.rotate(count=3)


class FillRainbow(_ImageTransform):
    @staticmethod
    def tag():
        return 'rainbow'

    def __call__(self):
        rainbows = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        colors = [numpy.array(matplotlib.colors.to_rgb(c)) * 255 for c in rainbows]
        mask = self.image.sum(axis=2) != 255 * 4
        for row in range(self.size):
            self.image[row, mask[row, :], :3] = colors[row % len(colors)]
        return self.image


def _select(tag: str, **kwargs):
    all_ = [Circle, Triangle, Square, Triangle, Square, Rectangle, Rhombus, Star, Hexagon, Crescent,
            Fix, DitherHalftone, DitherShaded, RotateClockwise, RotateCounterClockwise, RotateReverse,
            FillRainbow]
    for call in all_:
        if tag == call.tag():
            return call(**kwargs)
    raise ValueError(f"Unknown tag = {tag}")


class SampleMaker:
    RAINBOW_COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    FULL_COLORS = RAINBOW_COLORS + ['cyan', 'saddlebrown', 'black', 'gray', 'rainbow']
    SIMPLE_SHAPES = ['circle', 'triangle', 'square', 'rhombus', 'rectangle']
    FULL_SHAPES = SIMPLE_SHAPES + ['star', 'hexagon', 'crescent']
    FULL_SCALES = ['big', 'bigger', 'smaller', 'small']
    _DITHERS = ['', 'shaded', 'halftone']
    _FILLS = ['', 'filled']
    _ROTATES = ['', 'clockwise', 'reverse', 'counterclockwise']

    def __init__(self,
                 size: int,
                 colors: list = None,
                 shapes: list = None,
                 scales: list = None,
                 fill=True,
                 dither=True,
                 rotation=True):
        self._images = []
        self._labels = []
        self.size = size
        # Init parameters
        self.params = {
            'shape': shapes if isinstance(shapes, list) else self.FULL_SHAPES,
            'color': colors if isinstance(colors, list) else self.FULL_COLORS,
            'scale': scales if isinstance(scales, list) else self.FULL_SCALES
        }
        if fill:
            self.params['fill'] = self._FILLS
        if dither:
            self.params['dither'] = self._DITHERS
        if rotation:
            self.params['rotation'] = self._ROTATES

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def shake(self, num_sample: int):
        bar = tqdm(range(num_sample), desc="shake samples")
        for _ in bar:
            param = {key: numpy.random.choice(value) for key, value in self.params.items()}
            self._labels.append(list(param.values()))
            self._images.append(self.__generate__(**param))

    def __generate__(self, shape: str, color: str, scale: str, fill=None, dither=None, rotation=None):
        image = numpy.ones((self.size, self.size, 4), dtype=numpy.uint8)
        shape = _select(shape, image=image, size=self.size, scale=scale, color=color)
        assert isinstance(shape, _CairoShape)
        shape()
        # Transform the shape with options
        if fill == 'filled':
            shape.fill()
        else:
            shape.stroke()
        if isinstance(rotation, str):
            rotation = _select(rotation, image=image, size=self.size)
            image = rotation()
        if isinstance(dither, str):
            dither = _select(dither, image=image, size=self.size)
            dither()
        # Fill the rainbow color
        if color == 'rainbow':
            filler = _select(color, image=image, size=self.size)
            filler()
        return image

    def show(self, figure_size=(10, 10), verbose=True):
        rows = int(math.sqrt(len(self._images)))
        cols = int(len(self._images) / rows)
        if rows * cols > len(self._images):
            return
        total_images = []
        for j in range(rows):
            im_row = []
            lb_row = []
            for i in range(cols):
                pos = j * cols + i
                im_row.append(self._images[pos])
                _labels = self._labels[j * cols + i]
                lb_row.append('_'.join(lb for lb in _labels if lb != ''))
            total_images.append(numpy.concatenate(im_row, axis=1))
            if verbose:
                print(';'.join(lb_row))
        total_images = numpy.concatenate(total_images)
        matplotlib.pyplot.figure(figsize=figure_size)
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.imshow(total_images)
        matplotlib.pyplot.show()

    def save(self, root_path: str, init_path=True):
        if init_path and os.path.exists(root_path):
            shutil.rmtree(root_path)
        os.makedirs(root_path, exist_ok=True)
        for im, lb in zip(self.images, self.labels):
            file_name = '_'.join(t for t in lb if t != '')
            im = Image.fromarray(im)
            im.save(f"{root_path}/{file_name}.png")


if __name__ == "__main__":
    train_data = SampleMaker(size=128)
    train_data.shake(1000)
    train_data.show()
    train_data.save('data/train/all')
    test_data = SampleMaker(size=128)
    test_data.shake(200)
    test_data.show()
    test_data.save('data/test/all')
