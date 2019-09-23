import os
import io
import tarfile
from myelindl.core.utils.bucket import DataBuckets, TagIdGenerator
from myelindl.core.util import get_folder_size, sizeof_fmt

TYPE = [
    "file",
    "tf_saved_model",
    "pytorch",
    "tf_luminoth",
    "tf_model_zoo",
]
buckets = DataBuckets(name='models', id_generator=TagIdGenerator())


def format_model_tag(username, model_name, tag):
    return '{}/{}:{}'.format(username, model_name, tag)


def parse_model_tag(model_tag):
    username, model = model_tag.split('/')
    model_name, tag = model.split(':')
    return username, model_name, tag


def push(tag, description, model_dir, type):
    if not os.path.exists(model_dir):
        raise Exception('Folder not found {}'.format(model_dir))

    user, name, version  = parse_model_tag(tag)
    if exists(tag):
        buckets.update_bucket(
            id=tag,
            metadata={
                'name': name,
                'username': user,
                'tag': tag,
                'version': version,
                'size': get_folder_size(model_dir),
                'description':description,
                'type': type,
            },
            src_dir=model_dir,
            keep_src=False,
        )
    else:
        tag = buckets.create_bucket(
            metadata={
                'name': name,
                'username': user,
                'tag': tag,
                'version': version,
                'size': get_folder_size(model_dir),
                'description':description,
                'type': type,
            },
            src_dir=model_dir,
            keep_src=False,
        )

    return tag


def download(model_tag):
    metadata = buckets.get_bucket_metadata(model_tag)
    
    if not metadata:
        raise Exception('Model not found {}.'.format(modle_tag))
    
    bucket_dir = buckets.get_bucket_dir(model_tag)
    b = io.BytesIO()
    name = os.path.basename(bucket_dir)
    with tarfile.open(fileobj=b, mode='w:gz') as tar:
        tar.add(bucket_dir, arcname=name)
    return b.getvalue()

def delete(model_tag):
    return buckets.delete_bucket(model_tag)


def server(model_tag):
    pass


def list(**kwargs):
    return buckets.list_bucket(**kwargs)


def info(model_tag):
    return buckets.get_bucket_metadata(model_tag)


def exists(model_tag):
    return True if buckets.get_bucket_metadata(model_tag) else False
