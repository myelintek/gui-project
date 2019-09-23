from .classification_writer import ClassificationWriter # noqa
from .object_detection_writer import MLSteamObjectDetectionWriter
#from .object_detection_writer import MLSteamObjectDetecionWriter  # noqa
from .tfrecords_writer import ProgressHook
from .multilabel_writer import MultilabelWriter
from .lidar_writer import LidarImageWriter
from empty_writer import OnlyLabelsWriter

WRITERS = {
    'classification': ClassificationWriter,
    'coco': MLSteamObjectDetectionWriter,
    'csv': MLSteamObjectDetectionWriter,
    'flat': MLSteamObjectDetectionWriter,
    'imagenet': MLSteamObjectDetectionWriter,
    'openimages': MLSteamObjectDetectionWriter,
    'pascal': MLSteamObjectDetectionWriter,
    'taggerine': MLSteamObjectDetectionWriter,
    'kitti': MLSteamObjectDetectionWriter,
    'chestxray': MultilabelWriter,
    'lidar_npy': LidarImageWriter,
    'yolo': OnlyLabelsWriter,
    'segmentation': OnlyLabelsWriter,
}


def get_writer(writer):
    writer = writer.lower()
    if writer not in WRITERS:
        raise ValueError('"{}" is not a valid writer'.format(writer))
    return WRITERS[writer]
