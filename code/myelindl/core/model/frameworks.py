import os
import click
import requests
import tarfile
import json
import pprint
import shutil
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
import db_lib

from luminoth.tools.checkpoint import (
    download_remote_checkpoint,
    get_checkpoint,
    read_checkpoint_db,
    get_checkpoints_directory,
    save_checkpoint_db,
    CHECKPOINT_INDEX,
    CHECKPOINT_PATH,
    merge_index,
    get_checkpoint_config,
)
 
import uuid
import click
from PIL import Image
from six.moves import _thread
from luminoth.utils.config import get_config, override_config_params
from luminoth.utils.predicting import PredictorNetwork
from shutil import copyfile
from .torch_predicting import PredictorNetwork as TorchPredictorNetwork
from .torch_checkpoint import download_torch_remote_checkpoint

logger = logging.getLogger('myelindl.core.model.frameworks')

DEFAULT_SCORE_THRESHOLD = 0.1
OUTPUT_NAMES = ['labels', 'objects', 'scale_factor', 'probs' ]


class Framework(object):

    def inference(self, username, model, input_dir, score_threshold):
        pass

    def download(self, model, force):
        pass

    def create_inference_server(self, config):
        pass

    def get_model_convert_output(self, model):
        pass


class LuminothFramework(Framework):

    def inference(self, model, input_dir, score_threshold):
        config = get_checkpoint_config(network,prompt=False)
        if not config:
            return flask.jsonify({'error':'network not found'})
        try:
            predictor = PredictorNetwork(config)
            paths = os.listdir(input_dir) if os.path.isdir(input_dir) else [input_dir]
            for image_file in paths:
                image = Image.open(image_file).conver('RGB')
                predictor.predict_image(image)
        except Exception as e:
            # An error occurred loading the model; interrupt the whole server.
            _thread.interrupt_main()

    def download(self, network, force):
       db = db_lib.read_model_db()
       checkpoint = db_lib.get_model(db, network)
       if not checkpoint:
           click.echo(
               "Checkpoint '{}' not found in index.".format(network)
           )
           return

       if checkpoint['source'] != 'remote':
           click.echo(
               "Checkpoint is not remote. If you intended to download a remote "
               "checkpoint and used an alias, try using the id directly."
           )
           return

       if checkpoint['status'] != 'NOT_DOWNLOADED':
           click.echo("Checkpoint is already downloaded.")
           return

       download_remote_checkpoint(db, checkpoint) 

    def create_inference_server(self, config):
        return PredictorNetwork(config)

    def get_model_convert_output(self, model):
	
        config = get_checkpoint_config(model)
        network = PredictorNetwork(config)

        output_names = []
        for k, v in network.fetches.iteritems():
            if k in OUTPUT_NAMES:
                if isinstance(v, tf.Tensor):
                    output_names.append(v.name.split(':')[0]) 
                elif isinstance(v, tuple):
                    for x in v:
                        output_names.append(x.name.split(':')[0])
            
        logger.debug( 'Input name: {}\n'.format(network.image_placeholder.name))
	for n in output_names:
            logger.debug('Output name {}'.format(n))
	return network.session, network.session.graph_def, output_names


class PytorchFramework(Framework):
    def __init__(self):
       #self.user = user
       pass

    def download(self, model, force):
        ''' model is like id: 829766922537 '''
        db = db_lib.read_model_db()
        checkpoint = db_lib.get_model(db, model)
        if not checkpoint:
           click.echo(
               "Checkpoint '{}' not found in index.".format(model)
           )
           return
        if checkpoint['status'] != 'NOT_DOWNLOADED':
           click.echo("Checkpoint is already downloaded.")
           return

        download_torch_remote_checkpoint(db, checkpoint)


    def create_inference_server(self, config):
        return TorchPredictorNetwork(config)


    def get_model_convert_output(self, model):
        raise ValueError('Not support this function')

#class ModelZooFramework(Framework):
#
#    def __init__(self):
#
#        self.MODEL_ZOO_URL = 'http://download.tensorflow.org/models/object_detection/'
#
#        self.PRETRAINED_MODELS = {
#            'ssd_mobilenet_v2_coco': 'ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
#            'ssd_mobilenet_v1_coco': 'ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
#            'faster_rcnn_resnet101_kitti': 'faster_rcnn_resnet101_kitti_2018_01_28.tar.gz',
#        }
#
#        self.PRETRAINED_LABELS = {
#            'ssd_mobilenet_v2_coco': '/build/lib/models/research/object_detection/data/mscoco_label_map.pbtxt',
#            'ssd_mobilenet_v1_coco': '/build/lib/models/research/object_detection/data/mscoco_label_map.pbtxt',
#            'faster_rcnn_resnet101_kitti':'/build/lib/models/research/object_detection/data/kitti_label_map.pbtxt',
#        }
#        self.MODEL_DIR_NAME='model'
#        self.PRETRAINED_MODEL_DIR = os.path.join(config_value('jobs_dir'), 'pretrained_model')
#
#
#    def support_models(self):
#        return self.PRETRAINED_MODELS.keys()
#
#    def list(self):
#        for network in self.PRETRAINED_MODELS.keys():
#            pretrained_model_path = os.path.join(self.PRETRAINED_MODEL_DIR, network)
#            status = "downloaded" if os.path.exists(pretrained_model_path) else "N/A"
#            print('{}:{}'.format(network, status))
#
#    def download(self, network, force=False):
#        pretrained_model_path = os.path.join(self.PRETRAINED_MODEL_DIR, network)
#
#        if force and os.path.isdir(pretrained_model_path):
#            shutil.rmtree(pretrained_model_path)
#            
#
#        if os.path.exists(pretrained_model_path):
#            logger.warning('Model exists {}'.format(network))
#            return
#        else:
#            os.makedirs(pretrained_model_path)
#
#
#        tarball_filename = self.PRETRAINED_MODELS[network]
#        tarball_temp_path = os.path.join(self.PRETRAINED_MODEL_DIR, tarball_filename.split('.')[0])
#        url = self.MODEL_ZOO_URL + tarball_filename  
#
#        response = requests.get(url, stream=True)
#        total_size = int(response.headers.get('Content-Length'))
#        tarball_path = os.path.join(self.PRETRAINED_MODEL_DIR, tarball_filename)
#        tmp_tarball = tf.gfile.Open(tarball_path, 'wb')
#        tf.logging.info('Downloading {} checkpoint.'.format(tarball_filename))
#        with click.progressbar(length=total_size) as bar:
#            for data in response.iter_content(chunk_size=4096):
#                tmp_tarball.write(data)
#                bar.update(len(data))
#        tmp_tarball.flush()
#
#        tf.logging.info('Saving checkpoint to {}'.format(self.PRETRAINED_MODEL_DIR))
#        # Open saved tarball as readable binary
#        tmp_tarball = tf.gfile.Open(tarball_path, 'rb')
#        # Open tarfile object
#        tar_obj = tarfile.open(fileobj=tmp_tarball)
#
#        tar_obj.extractall(path=self.PRETRAINED_MODEL_DIR)
#
#        os.rename(tarball_temp_path, os.path.join(self.PRETRAINED_MODEL_DIR, pretrained_model_path))
#            
#        tmp_tarball.close()
#
#        # Remove temp tarball
#        tf.gfile.Remove(tarball_path)
#
#    def load_inference_graph(self, frozen_graph_path):
#        graph = tf.Graph()
#        with graph.as_default():
#            gdef = tf.GraphDef()
#            with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
#                serialized_graph = fid.read()
#                gdef.ParseFromString(serialized_graph)
#                tf.import_graph_def(gdef, name='')
#
#        return graph
#
#    def load_image_into_numpy_array(self, image):
#        (im_width, im_height) = image.size
#        return np.array(image.getdata()).reshape(
#            (im_height, im_width, 3)).astype(np.uint8)
#
#    def run_inference_for_single_image(self, image, graph):
#
#        with graph.as_default():
#            with tf.Session() as sess:
#                # Get handles to input and output tensors
#                ops = tf.get_default_graph().get_operations()
#                all_tensor_names = {output.name for op in ops for output in op.outputs}
#                tensor_dict = {}
#                for key in [
#                    'num_detections', 'detection_boxes', 'detection_scores',
#                    'detection_classes', 'detection_masks'
#                ]:
#                    tensor_name = key + ':0'
#                    if tensor_name in all_tensor_names:
#                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
#                            tensor_name)
#                if 'detection_masks' in tensor_dict:
#                    # The following processing is only for single image
#                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
#                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
#                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
#                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
#                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
#                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
#                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
#                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
#                    detection_masks_reframed = tf.cast(
#                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
#                    # Follow the convention by adding back the batch dimension
#                    tensor_dict['detection_masks'] = tf.expand_dims(
#                        detection_masks_reframed, 0)
#                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
#      
#                # Run inference
#                output_dict = sess.run(tensor_dict,
#                                       feed_dict={image_tensor: np.expand_dims(image, 0)})
#      
#                # all outputs are float32 numpy arrays, so convert types as appropriate
#                output_dict['num_detections'] = int(output_dict['num_detections'][0])
#                output_dict['detection_classes'] = output_dict[
#                    'detection_classes'][0].astype(np.uint8)
#                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#                output_dict['detection_scores'] = output_dict['detection_scores'][0]
#                if 'detection_masks' in output_dict:
#                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
#        return output_dict
#
#
#    def ensure_jpeg(self, image):
#        if image.format == 'JPEG':
#            return image
#
#        return image.convert('RGB')
#
#
#    def inference(self, network, input_dir, score_threshold):
#      
#        paths = os.listdir(input_dir) if os.path.isdir(input_dir) else [input_dir]
#        detection_graph = self.load_inference_graph(os.path.join(self.PRETRAINED_MODEL_DIR, network, 'frozen_inference_graph.pb'))
#        category_index = label_map_util.create_category_index_from_labelmap(self.PRETRAINED_LABELS[network], use_display_name=True)
#        results={}
#        for image_path in paths:
#            #try:
#            image = self.ensure_jpeg(Image.open(image_path))
#            width, height = image.size
#            # the array based representation of the image will be used later in order to prepare the
#            # result image with boxes and labels on it.
#            image_np = self.load_image_into_numpy_array(image)
#            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#            image_np_expanded = np.expand_dims(image_np, axis=0)
#            # Actual detection.
#            
#            output_dict = self.run_inference_for_single_image(image_np, detection_graph)
#            # Visualization of the results of a detection.
#
#            object_score_threshold = score_threshold if score_threshold else DEFAULT_SCORE_THRESHOLD
#            #except:
#            #    tf.logging.error('Inference Error on {}'.format(image_path))
#            #    continue
#
#            results[image_path] = []
#            for label, box, score in zip (output_dict['detection_classes'], output_dict['detection_boxes'], output_dict['detection_scores']):
#                if score >= object_score_threshold:
#                    [ymin, xmin, ymax, xmax] = box.tolist()
#                    [ymin, xmin, ymax, xmax] = [ymin * height, xmin * width, ymax * height, xmax * width]
#                    results[image_path].append(
#                            {'label': category_index[label],
#                                'box':
#                                    {'ymin': ymin,
#                                    'xmin': xmin,
#                                    'ymax': ymax,
#                                    'xmax': xmax},
#                                'score': score.astype(float) })
#
#
#        print(json.dumps(results, indent=4 ))

DEFAULT_FRAMEWORK= 'luminoth'

MODEL_FRAMEWORKS = {
#    'model_zoo': ModelZooFramework,
    'luminoth': LuminothFramework,
    'pytorch': PytorchFramework,
}


class FrameworkManager(object):
    def __init__(self):
        pass

    def list_models(self):
        db = db_lib.read_model_db()
        if not db['checkpoints']:
            raise ValueError('No model available.')
        return db['checkpoints']

    def get_framework_u(self, username, model):
        db = db_lib.read_model_db(username)
        model_db = db_lib.get_model(db, model)
        return MODEL_FRAMEWORKS[model_db['framework']]()
 
    def get_framework(self, model):
        db = db_lib.read_model_db()
        model_db = db_lib.get_model(db, model)

        return MODEL_FRAMEWORKS[model_db['framework']]()
    
    def support_models(self):
        db = db_lib.read_model_db()
        return [ x['id'] for x in db['checkpoints']]

    def inference(self, username, model, input_dir, score_threshold):
        return self.get_framework( model).inference(username, model, input_dir, score_threshold)        

    def get_model_convert_output(self, model):
        return self.get_framework(model).get_model_convert_output(model)

    def download(self, model, force):
        self.get_framework(model).download(model, force)

    def create_server(self, username, network):
        from myelindl.core.model.server import inference_server_pool
        config = db_lib.get_model_config(username, network, False)
        framework = self.get_framework_u(username, network)
        # Bounding boxes will be filtered by frontend (using slider), so we set a
        # low threshold.
        if config is not None:
            if config.model.type == 'fasterrcnn':
                config.model.rcnn.proposals.min_prob_threshold = 0.01
            elif config.model.type == 'ssd':
                config.model.proposals.min_prob_threshold = 0.01
            #else:
                #config.model.proposals.min_prob_threshold = 0.01
        return inference_server_pool.create_inference_server(
                    username, framework, config)

    def stop_server(self, username, server_id):
        from myelindl.core.model.server import inference_server_pool
        inference_server_pool.del_inference_server(username, server_id)
 
    def list_server(self, username):
        from myelindl.core.model.server import inference_server_pool
        return {'models': inference_server_pool.get_inference_servers(username)}
    
    def server_inference(self, username, server_id, image_array, total_predictions):
        from myelindl.core.model.server import inference_server_pool
        server = inference_server_pool.get_inference_server(username, server_id) 
        objects = server.predict_image(image_array)
        objects = objects[:total_predictions]
        return objects


