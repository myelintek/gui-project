import numpy as np
from os import listdir
import skimage.transform
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import skimage
from skimage.io import *
from skimage.transform import *

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches
import copy
import db_lib
import getpass
from PIL import Image
import json
import logging

logger = logging.getLogger('myelindl.core.model.torch_predicting')

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# model archi
# construct model
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


# ======= build test dataset =======
class ChestXrayDataSet_plot(Dataset):
    def __init__(self, input_X, transform=None):
        self.X = np.uint8(input_X*255)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item 
        Returns:
            image 
        """
        current_X = np.tile(self.X[index],3)
        image = self.transform(current_X)
        return image
    def __len__(self):
        return len(self.X)


# ======= Grad CAM Function =========
class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
#         self.probs = F.softmax(self.preds)[0]
#         self.prob, self.idx = self.preds[0].data.sort(0, True)
        return self.preds.cpu().data.numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data[0]

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_()

        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data
        
        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))

    def save_raw(self, filename, gcam, raw_image, label, color, left, upper, width, height):
        # draw a green rectangle to visualize the bounding rect
        cv2.putText(raw_image, label, (left, upper), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        cv2.rectangle(raw_image, (left, upper), (left+width, upper+height), color, 3)
        logger.debug("...bbox:{}".format( (left, upper, left+width, upper+height) ))
        cv2.imwrite(filename, np.uint8(raw_image))


class PredictorNetwork(object):
    def __init__(self, config):
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        
        self.output_path = config.dataset.dir
        for fn in os.listdir(self.output_path):
            name , ext = os.path.splitext(fn)
            if ext =='.pkl':
                self.pkl_fname = fn
            elif ext == '.npy':
                self.npy_fname = fn
        
        self.model_pkl_path = os.path.join(self.output_path, self.pkl_fname)
        self.thresholds_path = os.path.join(self.output_path, self.npy_fname)

        self.class_index = ['Atelectasis', 'Cardiomegaly', 
                            'Effusion', 'Infiltration', 
                            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
        self.class_color_map = {'Atelectasis': (255, 0, 0),
                                'Cardiomegaly': (0, 255, 0),
                                'Effusion':(0, 0, 255),
                                'Infiltration':(255, 255, 0),
                                'Mass':(255, 0, 255),
                                'Nodule':(0, 255, 255),
                                'Pneumonia':(255, 128, 128),
                                'Pneumothorax':(128, 255, 128)}
        self.target_layer = "module.densenet121.features.denseblock4.denselayer16.conv.2"

        # ======= prepare the model and gcam object =========
        model = DenseNet121(len(self.class_index)).cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(self.model_pkl_path))
        logger.debug("model loaded")

        self.gcam = GradCAM(model=model, cuda=True)


    def predict_image(self, image):
        image_path = '/tmp/input.png'
        cv2.imwrite(image_path, np.array(image))
        #if not os.path.exists(image):
        #    sys.exit()
       
        test_X = []
        test_raw_X = []
        test_origin_X = []
        logger.debug("load a image and transform image")
        img = scipy.misc.imread(image_path)
        if img.shape != (1024,1024):
            img = img[:,:,0]
        img_resized = skimage.transform.resize(img,(256,256))
        test_X.append((np.array(img_resized)).reshape(256,256,1))
         
        #Danny Implementation
        img = cv2.imread(image_path)

        test_origin_X.append(img)
        test_raw_X.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
        
        test_X = np.array(test_X)
        test_raw_X = np.array(test_raw_X)
        
        test_dataset = ChestXrayDataSet_plot(input_X = test_X,transform=transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])]))
                
        thresholds = np.load(self.thresholds_path)
        logger.debug("activate threshold {}".format(thresholds))
        logger.debug("generate heatmap ..........")
        
        # ======== Create heatmap ===========
        heatmap_output = []
        image_id = []
        output_class = []
        
        for index in range(len(test_dataset)):
            input_img = Variable((test_dataset[index]).unsqueeze(0).cuda(), requires_grad=True)
            probs = self.gcam.forward(input_img)
        
            activate_classes = np.where((probs > thresholds)[0]==True)[0] # get the activated class
            for activate_class in activate_classes:
                self.gcam.backward(idx=activate_class)
                output = self.gcam.generate(
                            target_layer=self.target_layer)
                
                # Danny Implementation
                file_name = os.path.join('/tmp', 
                    str(index) + '_heatmap_' + self.class_index[activate_class] + '.png')
                self.gcam.save(file_name, output, test_raw_X[index])
                #### this output is heatmap ####
                if np.sum(np.isnan(output)) > 0:
                    logger.debug("Got Nan in output")
                heatmap_output.append(output)
                image_id.append(index)
                output_class.append([activate_class, probs[index][activate_class]])
                
            logger.debug("The image to do inferencing {} is finished".format(str(index)))
        logger.debug("heatmap output done")
        logger.debug("total number of heatmap: {}".format(len(heatmap_output)))
        
        # ======= Plot bounding box =========
        img_width, img_height = 224, 224
        img_width_exp, img_height_exp = 1024, 1024
        
        crop_del = 16
        rescale_factor = 4
        avg_size = np.array([[411.8, 512.5, 219.0, 139.1], [348.5, 392.3, 479.8, 381.1],
                             [396.5, 415.8, 221.6, 318.0], [394.5, 389.1, 294.0, 297.4],
                             [434.3, 366.7, 168.7, 189.8], [502.4, 458.7, 71.9, 70.4],
                             [378.7, 416.7, 276.5, 304.5], [369.3, 209.4, 198.9, 246.0]])
        
        
        prediction_dict = {}
        prediction_dict[0] = []

        # currently we only handle one input image
        objects_list = []

        for img_id, k_and_prob, npy in zip(image_id, output_class, heatmap_output):
            k = k_and_prob[0]
            prob = k_and_prob[1]

            data = npy
        
            # output avgerge
            prediction_sent = '%s %.1f %.1f %.1f %.1f' % (self.class_index[k], 
                                                          avg_size[k][0], 
                                                          avg_size[k][1], 
                                                          avg_size[k][2], 
                                                          avg_size[k][3])
            prediction_dict[img_id].append(prediction_sent)
        
            if np.isnan(data).any():
                continue
        
            w_k, h_k = (avg_size[k][2:4] * (256.0 / 1024.0)).astype(np.int)
            
            # Find local maxima
            neighborhood_size = 100
            threshold = .1
            
            data_max = filters.maximum_filter(data, neighborhood_size)
            maxima = (data == data_max)
            data_min = filters.minimum_filter(data, neighborhood_size)
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 0
            
            for _ in range(5):
                maxima = binary_dilation(maxima)
            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)
            xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
            
            seq_num = 0
            for pt in xy:
                if data[int(pt[0]), int(pt[1])] > np.max(data)*.9:
                    seq_num += 1
                    upper = int(max(pt[0]-(h_k/2.0), 0.))
                    left = int(max(pt[1]-(w_k/2.0), 0.))
        
                    right = int(min(left+w_k, img_width))
                    lower = int(min(upper+h_k, img_height))
                    
                    prediction_sent = '%s %.1f %.1f %.1f %.1f' % (self.class_index[k], (left+crop_del)*rescale_factor, \
                                                                     (upper+crop_del)*rescale_factor, \
                                                                     (right-left)*rescale_factor, \
                                                                     (lower-upper)*rescale_factor)
                    prediction_dict[img_id].append(prediction_sent)
        
                    file_name = os.path.join('/tmp', 
                        str(img_id) + '_bb_' + self.class_index[k] + '_' +  str(seq_num) + '.png')
                    # OpenCV's drawing function will directly modify the data
                    # so that it needs to pass with copy of image 
                    # if you are want to accumulate the result of drawing
                    copy_img = copy.copy(test_origin_X[img_id])
                    self.gcam.save_raw(file_name, heatmap_output[img_id], copy_img, \
                               self.class_index[k], self.class_color_map[self.class_index[k]], \
                               (left+crop_del)*rescale_factor, \
                               (upper+crop_del)*rescale_factor, \
                               (right-left)*rescale_factor, \
                               (lower-upper)*rescale_factor)
                    object_data = {}
                    object_data["bbox"] = [(left+crop_del)*rescale_factor, \
                               (upper+crop_del)*rescale_factor, \
                               (left+crop_del)*rescale_factor + (right-left)*rescale_factor, \
                               (upper+crop_del)*rescale_factor + (lower-upper)*rescale_factor]
                    object_data["label"] = self.class_index[k]
                    object_data["prob"] = round(prob, 4)
                    objects_list.append(object_data)
        return objects_list
 
        with open("bounding_box.txt","w") as f:
            for i in range(len(prediction_dict)):
                prediction = prediction_dict[i]
        
                logger.debug("{} {}".format(os.path.join(image_path), len(prediction)))
                f.write('%s %d\n' % (os.path.join(image_path), len(prediction)))
        
                for p in prediction:
                    logger.debug(p)
                    f.write(p+"\n")

    
if __name__ == '__main__':
    img_file = sys.argv[1]
    img = Image.open(img_file)
    logger.debug("load a image and transform image")
    predict = PredictorNetwork("config")
    logger.debug("Result of json:", predict.predict_image(img))


