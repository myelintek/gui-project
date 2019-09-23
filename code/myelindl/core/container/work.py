import os
import re
import uuid
import time
import pwd
import signal
import subprocess
from myelindl import utils
from myelindl.log import logger
from myelindl.utils.container import get_real_path
from myelindl.webapp import scheduler
from myelindl.core.run import Run, RUN
from myelindl.env import *
from myelindl.core.const import DEFAULT_IMG
from myelindl.core.dataset import bucket
from myelindl.core.container.image import Image
import traceback

WRK_PREFIX = 'wrk-'
IDLE_TIMEOUT = 600 # 10 minutes


class Work(Run):
    def __init__(self, username, container, port_list, num_gpu, dataset='', project='', user_args=None):
        id = RUNS_PREFIX + WRK_PREFIX + str(uuid.uuid4()).replace('-', '')[:8]
        super(Work, self).__init__(id, username, 'work', status_history=[])
        self.port_list = port_list
        self.num_gpu = int(num_gpu)
        self.container = container
        self.dataset_path = bucket.get_dataset_path(dataset)
        self.project = project
        self.user_args = user_args
        self.persistent = False
        self.status = INIT
        scheduler.add_instance(self)

    def _dict_cust(self, d):
        if 'p' in d:
            del d['p']
        if 'current_resources' in d:
            del d['current_resources']
        return d

    def offer_resources(self, resources):
        port_identifiers = []
        if self.port_list:
            for resource in resources['ports']:
                if resource.remaining() >= 1:
                    # check for port availability
                    cmd = 'docker run --rm -p {}:8888 {} bash'.format(resource.identifier, DEFAULT_IMG)
                    ret = os.system(cmd)
                    if ret == 0:
                        port_identifiers.append(resource.identifier)
                if len(port_identifiers) == len(self.port_list):
                    break
        logger.debug("port_identifiers: {}".format(port_identifiers))

        gpu_identifiers = []
        if self.num_gpu:
            for resource in resources['gpus']:
                if resource.remaining() >= 1:
                    gpu_identifiers.append(resource.identifier)
                if len(gpu_identifiers) == self.num_gpu:
                    break
        logger.debug("gpu_identifiers: {}".format(gpu_identifiers))

        if (len(port_identifiers) == len(self.port_list)) and \
           (len(gpu_identifiers) == self.num_gpu):
            logger.debug("return resources.. ")
            resources = {'gpus': [(i, 1) for i in gpu_identifiers],
                         'ports': [(i, 1) for i in port_identifiers],
                         'hosts': [(resources['hosts'][0].identifier, 1)]}
            self.resources = resources
            self.save()
            return resources
        else:
            return None

    def run(self, resources):
        logger.info("Run worker!")
        self.before_run()
        env = os.environ.copy()
        gpus = [ i for (i, _) in resources['gpus'] ]
        ports = [ i for (i, _) in resources['ports'] ]
        env['NV_GPU'] = ','.join(str(g) for g in gpus)
        args = ['/usr/bin/docker', 'create', '--runtime=nvidia', '--rm', '--name', self.id]
        args.extend(['-e', 'NVIDIA_VISIBLE_DEVICES='+env['NV_GPU']])
        #user_uid = pwd.getpwnam(self.username).pw_uid
        #args.extend(['-u', '{}:{}'.format(user_uid, user_uid)])
        if ports:
            for i, port in enumerate(ports):
                args.extend(['-p', "{}:{}".format(port, self.port_list[i])])
        if self.dataset_path:
            dataset_host_path = get_real_path(self.dataset_path)
            args.extend(['-v', dataset_host_path+':/dataset'])
        if self.user_args:
            for n, arg in enumerate(self.user_args):
                if "JOB_ID" in arg:
                    self.user_args[n] = arg.replace("JOB_ID", self.id)
        home_dir = os.path.join('/home/', self.username)
        home_real_path = get_real_path(home_dir)
        args.extend(['-v', home_real_path+':/workspace', '-w', '/workspace'])
        args.extend([self.container] + self.user_args)
        logger.info("Run {}".format(' '.join(args)))
        try:
            output = subprocess.check_output(args)
        except subprocess.CalledProcessError as exc:
            for line in exc.output.split('\n'):
                self.output_log.write('%s\n' % line)
            self.after_run()
            logger.debug("docker create: {}".format(exc))
            raise exc
        logger.info("Run output: {}".format(output))
        container_id = re.findall(r"^\w+", output)[0][:6]
        args = ["/usr/bin/docker", "start", "-i", container_id]
        self.p = subprocess.Popen(args,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  cwd=self._dir,
                                  close_fds=True,
                                  env=env,
                                  )
        start_time = time.time()
        self.idle_sec = 0
        try:
            sigterm_time = None  # When was the SIGTERM signal sent
            sigterm_timeout = 120  # When should the SIGKILL signal be sent
            while self.p.poll() is None:
                for line in utils.nonblocking_readlines_p(self.p):
                    if self.aborted.is_set():
                        if sigterm_time is None:
                            if container_id:
                                subprocess.check_output("/usr/bin/docker stop {}".format(container_id), shell=True)
                            # Attempt graceful shutdown
                            self.p.send_signal(signal.SIGTERM)
                            sigterm_time = time.time()
                            self.status = ABORT
                        break
                    try:
                        # check for connection & timeout
                        netst = subprocess.check_output("/usr/bin/docker exec {} netstat -nat 8888".format(container_id), shell=True)
                        if 'ESTABLISHED' in netst:
                            start_time = time.time()
                        idle_sec = time.time() - start_time
                        if idle_sec > 10:
                            self.idle_sec = idle_sec
                        if idle_sec > IDLE_TIMEOUT:
                            self.abort()
                            logger.info("jupyter {} timeout, terminating..".format(self.id))
                    except:
                        pass
                    if line is not None:
                        # Remove whitespace
                        line = line.strip().rstrip()

                    if line:
                        self.output_log.write('%s\n' % line.encode("utf-8"))
                        self.output_log.flush()
                    else:
                        time.sleep(0.05)
                if sigterm_time is not None and (time.time() - sigterm_time > sigterm_timeout):
                    self.p.send_signal(signal.SIGKILL)
                    logger.debug("Sent SIGKILL to task {}".format(self.name))
                time.sleep(0.01)
        except Exception as e:
            logger.debug("exception, {}, {}".format(e, traceback.format_exc()))
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
    works = scheduler.get_works()
    work_list = [wrk.dict() for wrk in works]
    return work_list


def create(username, container, num_gpu, dataset, project, port_list=[], user_args=None):
    args = []
    bucket.check_exist(dataset)
    Image().get(container)
    if user_args:
        args = user_args.split()
    logger.info("create container: {}".format([username, container, num_gpu,
                dataset, project, port_list, user_args]))
    ser = Work(username, container, port_list, num_gpu, dataset, project, args)
    id = ser.id
    return id


def commit(username, name, id):
    works = scheduler.get_works()
    w = None
    for _w in works:
        if _w.id == id:
            w = _w
            break
    if w is None:
        raise ValueError('{} not found'.format(id))
    if w.status == RUN:
        subprocess.check_output("/usr/bin/docker commit {} {}".format(id, name), shell=True)
    else:
        raise ValueError('status not running')


def abort(run_id):
    ser = scheduler.get_instance(run_id)
    ser.abort()
