
from image_classification import ClassiReader
from luminoth.tools.dataset.readers.object_detection import (
#from .object_detection import (
    COCOReader, CSVReader, FlatReader, ImageNetReader, OpenImagesReader,
    PascalVOCReader, TaggerineReader
)
from chestxray14 import ChestXray14Reader
from myelindl.core.dataset.readers.kitti import KittiReader
from lidar_npy import LidarNpyReader
from yolo import YoloReader
from segmentation import SegmentationReader
READERS = {
    'classification': ClassiReader,
    'coco': COCOReader,
    'csv': CSVReader,
    'flat': FlatReader,
    'imagenet': ImageNetReader,
    'openimages': OpenImagesReader,
    'pascal': PascalVOCReader,
    'taggerine': TaggerineReader,
    'kitti': KittiReader,
    'chestxray': ChestXray14Reader,
    'lidar_npy': LidarNpyReader,
    'yolo': YoloReader,
    'segmentation': SegmentationReader,
}


class InvalidDataDirectory(Exception):
    """
    Error raised when the chosen intput directory for the dataset is not valid.
    """


def get_reader(reader):
    reader = reader.lower()
    if reader not in READERS:
        raise ValueError('"{}" is not a valid reader'.format(reader))
    return READERS[reader]
