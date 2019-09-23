import os
import uuid

CACHE_DIR = '/data/img_cache'


def img_id(content_type):
    suffix = content_type.split('/')[1]
    return '{}_{}'.format(str(uuid.uuid1())[:8], suffix)


def ext_from_id(img_id):
    return img_id.split('_')[-1]


def cache_dir(img_id):
    return os.path.join(CACHE_DIR,
            img_id[0],
            img_id[1])


def put(image, content_type='image/png'):
    '''
    put image datat and get id
    '''
    id = img_id(content_type)
    c_dir = cache_dir(id)
    if not os.path.exists(c_dir):
        os.makedirs(c_dir)

    with open(os.path.join(c_dir,id), 'wb') as f:
        f.write(image)
    return id


def get(id, delete_after=False):
    '''
    get image data by id 
    '''
    filepath = os.path.join(cache_dir(id), id)
    if not os.path.exists(filepath):
        raise ValueError('Cache not exists {}'.format(id))
    image = None
    with open(filepath, 'rb') as f:
        image = f.read()
    if delete_after:
        os.remove(filepath)
    ext = ext_from_id(id)

    return 'image/%s' % ext, image
