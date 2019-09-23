import os
import json
import docker
import uuid
from myelindl.log import logger
from myelindl.env import RUNS_DIR, RUNS_PREFIX
from myelindl.core.run import INIT, WAIT, RUN, DONE, ABORT, ERROR, Run
from myelindl.webapp import app, scheduler

from .registry import get_registry

IMAGE_PREFIX = 'img-'

class ImagePuller(Run):
    def __init__(self, image_tag):
        id = RUNS_PREFIX + IMAGE_PREFIX + str(uuid.uuid4()).replace('-', '')[:8]
        super(ImagePuller, self).__init__(id, 'container', 'container')
        self.image_tag = image_tag
        self.persistent = False
        self.status_history = []
        scheduler.add_instance(self)

    def parse_tag(self, tag):
        tag_parts = tag.split('/')
        cnt = len(tag_parts)
        if cnt == 3:
            host = tag_parts[0]
            repo = tag_parts[1]
            image = tag_parts[2]
        elif cnt == 2:
            host = ''
            repo = tag_parts[0]
            image = tag_parts[1]
        elif cnt == 1:
            host = ''
            repo = ''
            image = tag_parts[0]
        return host, repo, image


    def run(self, resources):
        self.before_run()

        host, repo, image = self.parse_tag(self.image_tag)
        logger.info("Pulling {}... ".format(self.image_tag))
        try:
            cli = docker.APIClient(base_url='unix://var/run/docker.sock')
            with app.app_context():
                registry = get_registry(host)
                registry.login(cli)

            for line in cli.pull(self.image_tag, stream=True, decode=True):
                if "progressDetail" in line:
                    progress = line["progressDetail"]
                    if "current" in progress:
                        percentag = float(progress['current']) / float(progress['total'])
                        self.progress = percentag
        except Exception as e:
            logger.warning("failed to pull image, {}".format(e))
        logger.info("done.")
        self.after_run()
        self.status = DONE


class Image(object):
    def __init__(self):
        self.client = docker.from_env()

    def list(self):
        images = self.client.images.list()
        img_list = []
        # import downloading images
        p_images = scheduler.get_runs(id_prefix=IMAGE_PREFIX)
        for img in p_images:
            if img.status == RUN:
                img_list.append({"id": "", "tag": img.image_tag, "progress": img.progress})
        for img in images:
            sid = img.short_id[7:]
            tags = img.tags
            if not tags:
                continue
            if 'mlsteam' in tags[0]:
                continue
            img_list.append({"id": sid, "tag": tags[0], "progress": 1.})
        return img_list

    def get(self, name):
        try:
            image = self.client.images.get(name)
            return image
        except Exception as e:
            raise ValueError('docker image {} not found.'.format(name))


    def delete(self, id):
        self.client.images.remove(id)
        return {'status': 'ok'}

    def pull(self, tag):
        tags = tag.rsplit(':', 1)    
        if len(tags) < 2:
            tags.append("latest")
        image_name = ":".join(tags)
        logger.info("create ImagePuller for {}".format(image_name))
        ImagePuller(image_name)
        # self.client.images.pull(tags[0], tags[1])
        return {'status': 'ok'}
