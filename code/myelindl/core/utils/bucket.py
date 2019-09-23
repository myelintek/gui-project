import os
import json
import shutil
import tempfile
import sys
import uuid
from threading import Lock
from datetime import datetime

from myelindl.core.util import copytree


class TagIdGenerator(object):
    def new_uuid(self, buckets, metadata):
        return metadata['tag']
        
        
class UuidIdGenerator(object):
    def new_uuid(self, buckets, metadata):
        while True:
            id = str(uuid.uuid1())[:8]
            if not os.path.exists(buckets.get_bucket_dir(id)):
                return str(uuid.uuid1())[:8]

class DataBuckets(object):
    def __init__(self, name, basedir='/data', id_generator=UuidIdGenerator()):
        self.name = name
        self.basedir = basedir
        self.bucket_dir = os.path.join(self.basedir, self.name)
        self.metadata_file = os.path.join(self.bucket_dir, 'metadata.json')
        self.metadata_lock = Lock()
        self.ensure_bucket_dir()
        self.id_generator = id_generator

    def get_bucket_dir(self, id):
        return os.path.join(self.bucket_dir, id)
    
    def list_bucket(self, **kwargs):
        data = self.read_metadata()
        offset = kwargs.pop("offset", 0)
        limit = kwargs.pop("limit", None)
        
        def filter_item(bucket):
            return all(map(lambda item: item[0] in bucket and bucket[item[0]] == item[1], kwargs.items()))
        
        result = filter(filter_item, data.values()) if kwargs else data.values()
        return result[offset:limit]

    def get_bucket_metadata(self, id):
        data = self.read_metadata()
        if id not in data:
            return {}
        return data[id]
        
    def create_bucket(self, metadata, src_dir=None, keep_src=True):
        id = self.id_generator.new_uuid(self, metadata)
        try:
            if src_dir:
                bucket = self.get_bucket_dir(id) 
                if keep_src:
                    if not os.path.exists(bucket):
                        os.mkdir(bucket)
                    copytree(src_dir, bucket)
                else:
                    shutil.move(src_dir, bucket)
            with self.metadata_lock:
                data = self.read_metadata()
                current = str(datetime.now())
                metadata['create_time'] = current
                metadata['update_time'] = current
                data[id] = metadata
                self.save_metadata(data)
            return id
        except Exception as e:
            self.delete_bucket(id)
            raise e
        
    def update_bucket(self, id, metadata, src_dir=None, keep_src=True):
        try:
            tempdir=None
            if src_dir:
                bucket_dir = self.get_bucket_dir(id)
                tempdir = tempfile.mkdtemp()
                shutil.move(bucket_dir, tempdir)
                if keep_src:
                    os.mkdir(bucket_dir)
                    copytree(src_dir, bucket_dir)
                else:
                    shutil.move(src_dir, bucket_dir)

            with self.metadata_lock:
                data = self.read_metadata()
                current = str(datetime.now())
                metadata['update_time'] = current
                data[id].update(metadata)
                self.save_metadata(data)
            return id
        except Exception as e:
            if tempdir:
                shutil.move(tempdir, bucket_dir)
            raise e
        finally:
            if src_dir and tempdir:
                shutil.rmtree(tempdir)

    def delete_bucket(self, id):
        bucket_dir = self.get_bucket_dir(id)
        if os.path.exists(bucket_dir):
            shutil.rmtree(bucket_dir)
        with self.metadata_lock:
            data = self.read_metadata()
            if id in data:
               del data[id]
            self.save_metadata(data)

    def read_metadata(self):
        if not os.path.exists(self.metadata_file):
            return {}
        data = {}
        with open(self.metadata_file, 'r') as infile:
            data = json.load(infile)
        return data

    def save_metadata(self, metadata):
        with open(self.metadata_file, 'w') as outfile:
            json.dump(metadata, outfile)

    def ensure_bucket_dir(self):
        if not os.path.exists(self.bucket_dir):
            os.makedirs(self.bucket_dir)
        
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as outfile:
                json.dump({}, outfile)

