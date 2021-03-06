#!/usr/bin/env python

import argparse
import glob
import io
import os
import random

import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2350-common-hangul.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../image-data')

DISTORTION_COUNT = 3

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32


def generate_hangul_images(label_file, fonts_dir, output_dir):

    with io.open(label_file, 'r', encoding='utf-8-sig') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'hangul-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
                         encoding='utf-8')

    total_count = 0
    prev_count = 0
    for character in labels:
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        for font in fonts:
            total_count += 1
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=150)
            font = ImageFont.truetype(font, 24)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)
            drawing.text(
                ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                character,
                fill=(10),
                font=font
            )
            file_string = 'hangul_{}.jpeg'.format(total_count)
            file_path = os.path.join(image_dir, file_string)
            image.save(file_path, 'JPEG')
            labels_csv.write(u'{},{}\n'.format(file_path, character))

            for i in range(DISTORTION_COUNT):
                total_count += 1
                file_string = 'hangul_{}.jpeg'.format(total_count)
                file_path = os.path.join(image_dir, file_string)
                arr = numpy.array(image)

                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )
                distorted_image = Image.fromarray(distorted_array)
                distorted_image.save(file_path, 'JPEG')
                labels_csv.write(u'{},{}\n'.format(file_path, character))

    print('Finished generating {} images.'.format(total_count))
    labels_csv.close()


def elastic_distort(image, alpha, sigma):

    random_state = numpy.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir)
