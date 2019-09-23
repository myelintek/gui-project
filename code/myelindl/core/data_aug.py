import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.python.layers import utils
from tensorflow.contrib.image.python.ops import distort_image_ops

R_BRIGHTNESS_MAX_DELTA = 32. / 255.
R_SATURATION_LOWER = 0.5
R_SATURATION_UPPER = 1.5
R_HUE_MAX_DELTA = 0.2
R_CONSTRAST_LOWER = 0.5
R_CONSTRAST_UPPER = 1.5

class ColorAugmentation(object):
    def __init__(self, img_operation):

        """ disable ranges to simplify function
        ranges = {
          "random_brightness": np.linspace(0, 1, 11),
          "random_saturation": (np.linspace(0, 1, 11), np.linspace(1, 2, 11)),
          "random_hue": np.linspace(0, 1, 11),
          "random_contrast": (np.linspace(0, 1, 11), np.linspace(1, 2, 11)),
        }
        """

        func = {
            "random_brightness": lambda img: tf.image.random_brightness(img,
                                                                        max_delta=R_BRIGHTNESS_MAX_DELTA),
            "random_saturation": lambda img: tf.image.random_saturation(img,
                                                                        lower=R_SATURATION_LOWER, upper=R_SATURATION_UPPER),
            "random_hue": lambda img: tf.image.random_hue(img,
                                                          max_delta=R_HUE_MAX_DELTA),
            "random_contrast": lambda img: tf.image.random_contrast(img,
                                                                    lower=R_CONSTRAST_LOWER, upper=R_CONSTRAST_UPPER),
        }

        self.img_operation = func[img_operation]
        #self.magnitude = ranges[img_operation][p1]

    def __call__(self, img):
        with tf.name_scope(name='ColorAugmentation'):
            return self.img_operation(img)

def distort_color(image, batch_position=0, distort_color_in_yiq=False):
    def distort_fn_0(image=image):
        """Variant 0 of distort function."""
        image = tf.image.random_brightness(image, max_delta=R_BRIGHTNESS_MAX_DELTA)
        if distort_color_in_yiq:
            image = distort_image_ops.random_hsv_in_yiq(
                image, lower_saturation=R_SATURATION_LOWER, upper_saturation=R_SATURATION_UPPER,
                max_delta_hue=R_HUE_MAX_DELTA * math.pi)
        else:
            image = tf.image.random_saturation(image, lower=R_SATURATION_LOWER, upper=R_SATURATION_UPPER)
            image = tf.image.random_hue(image, max_delta=R_HUE_MAX_DELTA)
        image = tf.image.random_contrast(image, lower=R_CONSTRAST_LOWER, upper=R_CONSTRAST_UPPER)
        return image
    def distort_fn_1(image=image):
        """Variant 1 of distort function."""
        image = tf.image.random_brightness(image, max_delta=R_BRIGHTNESS_MAX_DELTA)
        image = tf.image.random_contrast(image, lower=R_CONSTRAST_LOWER, upper=R_CONSTRAST_UPPER)
        if distort_color_in_yiq:
            image = distort_image_ops.random_hsv_in_yiq(
                image, lower_saturation=R_SATURATION_LOWER, upper_saturation=R_SATURATION_UPPER,
                max_delta_hue=R_HUE_MAX_DELTA * math.pi)
        else:
            image = tf.image.random_saturation(image, lower=R_SATURATION_LOWER, upper=R_SATURATION_UPPER)
            image = tf.image.random_hue(image, max_delta=R_HUE_MAX_DELTA)
        return image
    image = utils.smart_cond(batch_position % 2 == 0, distort_fn_0,
                             distort_fn_1)
    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

_RESIZE_METHOD_MAP = {
    'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    'bilinear': tf.image.ResizeMethod.BILINEAR,
    'bicubic': tf.image.ResizeMethod.BICUBIC,
    'area': tf.image.ResizeMethod.AREA
}

def get_image_resize_method(resize_method, batch_position=0):
    if resize_method != 'round_robin':
        return _RESIZE_METHOD_MAP[resize_method]
    # return a resize method based on batch position in a round-robin fashion.
    resize_methods = list(_RESIZE_METHOD_MAP.values())
    def lookup(index):
        return resize_methods[index]
    def resize_method_0():
        return utils.smart_cond(batch_position % len(resize_methods) == 0,
                                lambda: lookup(0), resize_method_1)
    def resize_method_1():
        return utils.smart_cond(batch_position % len(resize_methods) == 1,
                                lambda: lookup(1), resize_method_2)
    def resize_method_2():
        return utils.smart_cond(batch_position % len(resize_methods) == 2,
                                lambda: lookup(2), lambda: lookup(3))
    # NOTE(jsimsa): Unfortunately, we cannot use a single recursive function here
    # because TF would not be able to construct a finite graph.
    return resize_method_0()
