import os
import json
import sys
import zmq
import uuid
import time
import myelindl
import base64
import signal
import subprocess
from myelindl import utils
from myelindl.log import logger
from myelindl.webapp import scheduler
from myelindl.core.run import Run, RUN
from myelindl.env import *
from myelindl.core.checkpoint import get_checkpoint_path
SVC_PREFIX = 'svc-'

class Service(Run):
    def __init__(self, username, checkpoint_path):
        id = RUNS_PREFIX + SVC_PREFIX + str(uuid.uuid4()).replace('-', '')[:8]
        super(Service, self).__init__(id, username, 'service', status_history=[])
        self.checkpoint_path = checkpoint_path
        self.num_gpu = 1
        self.persistent = False
        scheduler.add_instance(self)
 
    def _dict_cust(self, d):
        if 'p' in d:
            del d['p']
        if 'current_resources' in d:
            del d['current_resources']
        return d

    def offer_resources(self, resources):
        if 'gpus' not in resources:
            return None
        identifiers = []
        for resource in resources['gpus']:
            if resource.remaining() >= 1:
                identifiers.append(resource.identifier)
                break
        if len(identifiers) == self.num_gpu:
            return {'gpus': [(i, 1) for i in identifiers]}
        else:
            return None

    def run(self, resources):
        self.before_run()
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(['.', self._dir, env.get('PYTHONPATH', '')] + sys.path)
        gpus = [ i for (i, _) in resources['gpus'] ]
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpus)
        root = os.path.dirname(os.path.abspath(myelindl.__file__))
        args = [sys.executable, '-m',
                os.path.join(root, 'tools', 'unix_server'),
                '--checkpoint-path=%s' % self.checkpoint_path,
                '--job-dir=%s' % self._dir,
                ]
        logger.debug("run args: {}".format(args))
        self.p = subprocess.Popen(args,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=self._dir,
                                  close_fds=True,
                                  env=env,
                                  )
        try:
            sigterm_time = None  # When was the SIGTERM signal sent
            sigterm_timeout = 120  # When should the SIGKILL signal be sent
            while self.p.poll() is None:
                for line in utils.nonblocking_readlines(self.p.stdout):
                # for line in self.p.stdout:
                    if self.aborted.is_set():
                        if sigterm_time is None:
                            # Attempt graceful shutdown
                            self.p.send_signal(signal.SIGTERM)
                            sigterm_time = time.time()
                            self.status = ABORT
                        break
                    if line is not None:
                        # Remove whitespace
                        line = line.strip()

                    if line:
                        self.output_log.write('%s\n' % line)
                        self.output_log.flush()
                    else:
                        time.sleep(0.05)
                if sigterm_time is not None and (time.time() - sigterm_time > sigterm_timeout):
                    self.p.send_signal(signal.SIGKILL)
                    logger.debug('Sent SIGKILL to task "%s"' % self.name)
                time.sleep(0.01)
        except Exception as e:
            logger.warning('service exception: {}'.format(e))
            self.p.terminate()
            self.after_run()
            raise e

        self.after_run()
        if self.status != RUN:
            return False
        if self.p.returncode != 0:
            self.returncode = self.p.returncode
            self.status = ERROR
        else:
            self.status = DONE
        return True


def list():
    services = scheduler.get_services()
    service_list = [svc.dict() for svc in services]
    return service_list


def create(username, checkpoint_id):
    checkpoint_path = get_checkpoint_path(checkpoint_id)
    logger.debug("get checkpoint_path: {}".format(checkpoint_path))
    svc = Service(username, checkpoint_path)
    id = svc.id
    return id


def abort(run_id):
    svc = scheduler.get_instance(run_id)
    svc.abort()


def inference(id, image):
    svc = scheduler.get_instance(id)
    # Connect to server by zmq
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        address = "ipc://{}/unix.socket".format(svc._dir)
        logger.debug("Connecting to {}".format(address))
        socket.connect(address)
        socket.send(image)
        # waiting for reply objects 
        message = socket.recv()
    except Exception as e:
        logger.warning('Inference fail: {}'.format(e))
        return []
    logger.debug("Reply: {}".format(message))
    return json.loads(message)
