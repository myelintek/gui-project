import os
import json
import tensorflow as tf

from PIL import Image
ROUND_NUM = 6
class PredictorNetwork(object):
    def __init__(self, config):
        self.session = tf.Session(graph=tf.Graph())
        
        if not os.path.exists(config.dataset.dir):
            raise ValueError('job folder not exists {}'.format(config.dataset.dir))

	self.predict = tf.contrib.predictor.from_saved_model(config.dataset.dir)
        with open(os.path.join(config.dataset.dir, 'classes.json'), 'r') as f:
            self.classes = json.load(f)
            

    def decode_image(self, bytearr):
        return bytearr

    def post_process(self, output):
        classes = [self.classes[int(c)-1] for c in output['classes'].tolist()]
        probs = output['probabilities'].tolist()

        probabilities = []
        for prob in probs:
            probability = []
            for i, p in enumerate(prob):
                probability.append((self.classes[i-1], round(p, ROUND_NUM)))

            probability = sorted(probability, key=lambda x : x[1], reverse=True)
            probabilities.append(probability) 
        return {
            'classes': classes,
            'probabilities': probabilities,
        }

    def predict_image(self, images):
        if not isinstance(images, list) and not isinstance(images, tuple):
            images = [images]
        predictions = self.predict({'image_bytes': images}) 
        return self.post_process(predictions)

        
    

if __name__ == '__main__':
   
    import base64
    from io import BytesIO
    from easydict import EasyDict as edict

    config = edict({
        'dataset':{
            'dir': '/data/checkpoints/1538687370/'
        }
    })
    predictor = PredictorNetwork(config)

    with open('/mnt/coco/val2017/000000039914.jpg','rb') as f:
        image = f.read()
    print type(image)
    print "===" *20
    print predictor.predict_image(image)
    print "===" *20
    print predictor.predict_image(image)
    print "===" *20
    print predictor.predict_image(image)
    print "===" *20
    print predictor.predict_image([image, image])
    print "===" *20
    print predictor.predict_image((image, image))


    img64 = base64.b64encode(image)
    print type(img64)
    image = base64.b64decode(img64)
    
    print type(image)
    print predictor.predict_image(image)
