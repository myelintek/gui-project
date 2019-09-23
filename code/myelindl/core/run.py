import os
import sys
import os.path
import json
import shutil
import uuid
import traceback
import subprocess
import signal
import time
import tarfile
import io
import re
import pwd
import time
import shutil
import docker
import gevent.event
from easydict import EasyDict
from myelindl.log import logger
from myelindl.env import *
from myelindl.utils.container import get_real_path

LOGFILE = 'output.log'

class Run(object):
    SAVE_INFO = 'info.json'

    @classmethod
    def load(cls, id):
        from myelindl.core.job.instance import JOBS_PREFIX, Instance
        from myelindl.core.container.image import IMAGE_PREFIX, ImagePuller
        _dir = os.path.join(RUNS_DIR, id)
        filename = os.path.join(_dir, cls.SAVE_INFO)
        with open(filename, 'rb') as savefile:
            j = EasyDict(json.load(savefile))
            run = None
            if JOBS_PREFIX in id:
                run = Instance(**j)
            elif IMAGE_PREFIX in id:
                run = ImagePuller(**j)
            #else:
            #    run = cls(**j)
            return run

    def __init__(self, id, username, name, status_history=[], persistent=True, progress = 0.):
        self.id = id
        self.username = username
        self.name = name
        self.status_history = status_history if status_history else list()
        self._dir = os.path.join(RUNS_DIR, self.id)
        self.aborted = gevent.event.Event()
        self._progress = progress
        self.persistent = persistent
        try:
            if not os.path.exists(self._dir):
                os.mkdir(self._dir)
                uid = pwd.getpwnam(self.username).pw_uid
                os.chown(self._dir, uid, uid)
            with open(os.path.join(self._dir, LOGFILE), 'a'):
                os.utime(os.path.join(self._dir, LOGFILE), None)
        except Exception as e:
            pass

    def path(self, filename, relative=False):
        if not filename:
            return None
        if os.path.isabs(filename):
            path = filename
        else:
            path = os.path.join(self._dir, filename)
        return str(path).replace("\\", "/")

    @property
    def _dict(self):
        d = self.__dict__.copy()
        if '_dir' in d:
            del d['_dir']
        if 'aborted' in d:
            del d['aborted']
        if 'output_log' in d:
            del d['output_log']
        if '_progress' in d:
            d['progress'] = d['_progress']
            del d['_progress']
        d = self._dict_cust(d)
        return d

    def _dict_cust(self, data):
        ### Implement by inherit class
        return data

    # Display to API
    def dict(self):
        data = self._dict
        if self.status == RUN:
            data['elapsed_time'] = time.time() - self.status_history[0][1]
        else:
            data['elapsed_time'] = self.status_history[-1][1] - self.status_history[0][1]
        return self.dict_cust(data)

    def dict_cust(self, data):
        ### Implement by inherit class
        return data

    def offer_resources(self, resources):
        ### return {} if use cpu only
        return {}
        ### return 'gpus' allocated resources
        return {'gpus': [(0, 1), (1, 1)]}
        ### return None if gpu is not avaiable
        return None

    def save(self):
        try:
            if not os.path.exists(self._dir):
                return
            tmpfile_path = self.path(self.SAVE_INFO + '.tmp')
            with open(tmpfile_path, 'wb') as tmpfile:
                data = json.dumps(self._dict,
                                  sort_keys=True,
                                  indent=4)
                tmpfile.write(data)
            file_path = self.path(self.SAVE_INFO)
            shutil.move(tmpfile_path, file_path)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.warning('Caught %s while saving run %s: %s' % (type(e).__name__, self.id, e))
            logger.debug(traceback.format_exc(e))
        return False

    @property
    def status(self):
        if len(self.status_history) > 0:
            return self.status_history[-1][0]
        return INIT

    @status.setter
    def status(self, new_status):
        if new_status not in [INIT, WAIT, RUN, DONE, ABORT, ERROR]:
            raise ValueError('set unknown status value: %s' % (new_status))

        if self.status_history and new_status == self.status_history[-1][0]:
            return
        now = time.time()
        self.status_history.append((new_status, now))
        self.save()
        self.on_status_update( new_status, now)

    def on_status_update(self, new_status, time):
        pass
    
    @property
    def progress(self):
        return self._progress
    
    @progress.setter
    def progress(self, progress):
        self._progress = progress
        self.on_progress_update(self._progress)

    def on_progress_update(self, progress):
        pass

    def download(self):
        b = io.BytesIO()
        path = self._dir
        name = os.path.basename(path)
        try:
            with tarfile.open(fileobj=b, mode='w') as tar:
                tar.add(path, arcname=name)
        except Exception as e:
            logger.error("tar fail, try tar per file")
            return self._tar_per_file(path)
        return b.getvalue()

    def _tar_per_file(self, fullpath):
        b = io.BytesIO()
        with tarfile.open(fileobj=b, mode="w") as tar:
            for root, dirs, files in os.walk(fullpath):
                for file in files:
                    try:
                        fpath = os.path.join(root, file)
                        tar.add(fpath)
                    except Exception as e:
                        logger.error('tar file failed, {}, {}'.format(file, e))
                        raise IOError('tar file failed, {}, {}'.format(file, e))
        return b.getvalue()

    def abort(self):
        """
        Abort the Task
        """
        if self.status == RUN or self.status == WAIT:
            self.aborted.set()
            self.status = ABORT

    def run(self, resources):
        ### Implement by inherit class
        pass

    def before_run(self):
        self.output_log = open(self.path(LOGFILE), 'a')

    def after_run(self):
        self.output_log.close()
        self.progress = 1.

    def delete(self):
        if self.status == RUN or self.status == WAIT:
            self.aborted.set()
            self.status = ABORT
        logger.info('Set abort {} and delete'.format(self.id))
        time.sleep(1)
        if os.path.exists(self._dir):
            shutil.rmtree(self._dir)
