import math

import numpy
import cairo
import matplotlib.pyplot
import matplotlib.colors
from cairo import Context
from numpy import ndarray


def _gen_circle(cr: Context):
    cr.arc(xc=0, yc=0, radius=1, angle1=0, angle2=2 * math.pi)


def _gen_triangle(cr: Context):
    cr.move_to(x=0, y=-1)
    cr.line_to(x=math.sqrt(3) / 2, y=0.5)
    cr.line_to(x=-math.sqrt(3) / 2, y=0.5)
    cr.line_to(x=-math.sqrt(3) / 2, y=0.5)
    cr.line_to(x=0, y=-1)


def _gen_square(cr: Context):
    cr.rectangle(x=-0.9, y=-0.9, width=1.8, height=1.8)


def _gen_rectangle(cr: Context):
    cr.rectangle(x=-0.9, y=-0.5, width=1.8, height=1)


def _gen_rhombus(cr: Context):
    cr.move_to(x=0, y=-1)
    cr.line_to(x=0.5, y=0)
    cr.line_to(x=0, y=1)
    cr.line_to(x=-0.5, y=0)
    cr.line_to(x=0, y=-1)


def _gen_star(cr: Context):
    cr.move_to(x=0, y=-1)
    cr.line_to(x=math.sqrt(3) / 2, y=0.5)
    cr.line_to(x=-math.sqrt(3) / 2, y=0.5)
    cr.line_to(x=0, y=-1)
    cr.move_to(x=0, y=1)
    cr.line_to(x=math.sqrt(3) / 2, y=-0.5)
    cr.line_to(x=-math.sqrt(3) / 2, y=-0.5)
    cr.line_to(x=0, y=1)


def _gen_haxagon(cr: Context):
    cr.move_to(x=0, y=-1)
    cr.line_to(x=math.sqrt(3) / 2, y=-0.5)
    cr.line_to(x=math.sqrt(3) / 2, y=0.5)
    cr.line_to(x=0, y=1)
    cr.line_to(x=-math.sqrt(3) / 2, y=0.5)
    cr.line_to(x=-math.sqrt(3) / 2, y=-0.5)
    cr.line_to(x=0, y=-1)


def _gen_crescent(cr: Context):
    cr.arc(xc=0, yc=0, radius=1, angle1=-math.pi * 0.5, angle2=math.pi * 0.5)
    cr.move_to(x=0, y=-1)
    cr.arc(xc=-math.sqrt(3), yc=0, radius=2, angle1=-math.pi / 6, angle2=math.pi / 6)


def _dither(mask: ndarray):
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


def _dither_solid():
    pass


def _dither_halftone(data: ndarray):
    mask = (data.astype(int).sum(axis=2) != 255 * 3).astype(float)
    mask *= 0.5
    _dither(mask)
    data[mask > 0.5, :] = 255


def _dither_shaded(data: ndarray):
    mask = (data.astype(int).sum(axis=2) != 255 * 3).astype(float)
    mask *= 0.3
    _dither(mask)
    data[mask > 0.5, :] = 255


class SampleMaker:
    RAINBOW_COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    FULL_COLORS = RAINBOW_COLORS + ['cyan', 'saddlebrown', 'black', 'gray', 'rainbow']
    SIMPLE_SHAPES = ['circle', 'triangle', 'square', 'rhombus', 'rectangle']
    FULL_SHAPES = SIMPLE_SHAPES + ['star', 'hexagon', 'crescent']
    FULL_SCALES = ['big', 'bigger', 'smaller', 'small']
    _ATTR_SHAPE = {'circle': _gen_circle, 'triangle': _gen_triangle, 'square': _gen_square, 'rhombus': _gen_rhombus,
                   'rectangle': _gen_rectangle, 'star': _gen_star, 'hexagon': _gen_haxagon}
    _ATTR_DITHER = {'': _dither_solid, 'shaded': _dither_shaded, 'halftone': _dither_halftone}
    _ATTR_SCALE = {'big': 1, 'bigger': 0.8, 'smaller': 0.6, 'small': 0.4}
    _ATTR_FILL = {'': False, 'filled': True}
    _ATTR_ROTATE = {'': 0, 'clockwise': 1, 'twice': 2, 'counterclockwise': 3}

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
            self.params['fill'] = ['', 'filled']
        if dither:
            self.params['ditherer'] = ['', 'shaded', 'halftone']
        if rotation:
            self.params['rotation'] = ['', 'clockwise', 'twice', 'counterclockwise']

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def shake(self, num_sample: int):
        for _ in range(num_sample):
            param = {key: numpy.random.choice(value) for key, value in self.params.items()}
            self._labels.append(list(param.values()))
            self._images.append(self.__generate__(**param))

    def __generate__(self, shape: str, color: str, scale: str, fill=None, dither=None, rotation=None):
        image = numpy.ones((self.size, self.size, 4), dtype=numpy.uint8)
        surface = cairo.ImageSurface.create_for_data(image, cairo.FORMAT_ARGB32, self.size, self.size)
        cr = cairo.Context(surface)
        cr.set_antialias(cairo.ANTIALIAS_NONE)
        cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        # Make the canvas
        cr.rectangle(0, 0, self.size, self.size)
        cr.set_source_rgb(1, 1, 1)
        cr.fill()
        # Draw the shape
        cr.set_line_width(1 / (self._ATTR_SCALE[scale] * self.size / 2))
        if color == 'rainbow':
            cr.set_source_rgb(0, 0, 0)
        else:
            cr.set_source_rgb(*reversed(matplotlib.colors.to_rgba(color)))
        self._ATTR_SHAPE[shape](cr)
        # Transform the shape with options
        if isinstance(fill, str) and self._ATTR_FILL[fill]:
            cr.stroke_preserve()
            cr.fill()
        else:
            cr.stroke()
        if isinstance(rotation, str):
            for _ in range(self._ATTR_ROTATE[rotation]):
                image = numpy.rot90(image)
        if isinstance(dither, str):
            self._ATTR_DITHER[dither](image)
        if color == 'rainbow':
            rainbow_rgb = [numpy.array(matplotlib.colors.to_rgb(c)) * 255 for c in self.RAINBOW_COLORS]
            mask = image.sum(axis=2) != 255 * 4
            for row in range(self.size):
                image[row, mask[row, :], :3] = rainbow_rgb[row % len(rainbow_rgb)]
        return image

    def show(self, shape: (int, tuple)):
        pass

    def save(self, root_path: str):
        pass
