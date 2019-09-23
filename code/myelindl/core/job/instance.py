import os
import sys
import os.path
import copy
import shutil
import uuid
import traceback
import subprocess
import signal
import time
import yaml
import pwd
import re
import docker
import myelindl
from myelindl.log import logger
from myelindl.utils.container import get_real_path
from myelindl.core.run import INIT, WAIT, RUN, DONE, ABORT, ERROR, Run
from myelindl.env import RUNS_DIR, RUNS_PREFIX
from myelindl.core import gogsop
from myelindl import utils
from myelindl.webapp import scheduler
from myelindl.core.util import copytree
from myelindl.core.const import DEFAULT_IMG
from subprocess import check_output
import traceback


JOBS_PREFIX = 'job-'


class Instance(Run):
    METRICS = 'metrics.csv'
    PARAMS  = 'param.yml'

    def __init__(self, id, username, name,
                 image_tag=None, dataset_path=None, module_name=None,
                 user_args=[], num_gpu=1, project=None,
                 status_history=[], persistent=True, progress=0.0, returncode=0,
                 sha=None, repo_path=None, parameters=None,
                 parent=None, child=[]):
        super(Instance, self).__init__(id, username, name,
            status_history, persistent, progress)
        self.image_tag = DEFAULT_IMG if image_tag is None else image_tag
        self.dataset_path = dataset_path
        self.num_gpu = int(num_gpu)
        self.user_args = user_args
        self.returncode = returncode
        self.sha = sha
        self.repo_path = repo_path
        self.parameters = parameters
        self.parent = parent
        self.child = child
        self._workspace = "/workspace"
        if project is None:
            self.project = self.name
        else:
            self.project = project

    def _dict_cust(self, d):
        if 'p' in d:
            del d['p']
        if 'current_resources' in d:
            del d['current_resources']
        if '_workspace' in d:
            del d['_workspace']
        return d

    def dict_cust(self, data):
        if data['user_args']:
            data['user_args'] = ' '.join(data['user_args'])
        if data['dataset_path']:
            data['dataset_path'] = re.sub('.*dataset/', 'bk/', data['dataset_path'])
        return data

    def offer_resources(self, resources):
        if 'gpus' not in resources:
            return None
        if self.num_gpu > 0:
            identifiers = []
            for resource in resources['gpus']:
                if resource.remaining() >= 1:
                    identifiers.append(resource.identifier)
                    if len(identifiers) == self.num_gpu:
                        break
            if len(identifiers) == self.num_gpu:
                resources = {'gpus': [(i, 1) for i in identifiers]}
                return resources
            else:
                return None
        return {}  # don't use a GPU at all

    def do_train(self):
        src_to = None
        repo_name = None
        if self.repo_path and (self.repo_path.startswith("ssh://") or self.repo_path.startswith("git@")):
            ## copy pkg from gogs
            logger.info("pkg clone from gogs or git")
            gogsop.repo_clone(self.username, self.repo_path, self._dir)
            repo_name = _repo_name(self.repo_path)
            src_to = os.path.join(self._dir, repo_name)
        elif self.repo_path and os.path.exists(self.repo_path):
            ## copy from another job
            logger.info("pkg clone from path {}".format(self.repo_path))
            repo_name = _repo_name(self.repo_path)
            src_to = os.path.join(self._dir, repo_name)
            system_copy(self.repo_path, src_to)
        elif self.repo_path and os.path.exists(os.path.join("/home", self.username, self.repo_path)):
            ## copy from workspace
            workspace = os.path.join("/home",self.username, self.repo_path)
            logger.info("pkg clone from workspace {}".format(workspace))
            repo_name = _repo_name(workspace)
            src_to = os.path.join(self._dir, repo_name)
            system_copy(workspace, src_to)
        elif self.repo_path:
            ## copy from CLI already
            logger.info("pkg from CLI {}".format(self.repo_path))
            repo_name = _repo_name(self.repo_path)
            if os.path.exists(os.path.join(self._dir, repo_name)):
                src_to = os.path.join(self._dir, repo_name)
        elif self.repo_path == '' and self.parent:
            ## copy from another job
            logger.info("replicate job {}".format(self.parent))
            job_dir = os.path.join('/data', 'dataset', self.parent, '')
            if os.path.exists(job_dir):
                rsync_copy(job_dir, self._dir)
        if src_to and os.path.isdir(src_to):
            self._workspace = "/workspace/{}".format(repo_name)
        logger.info('add instance {} to scheduler'.format(self.id))
        scheduler.add_instance(self)
 
    def run(self, resources):
        self.before_run()
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(['.', self._dir, env.get('PYTHONPATH', '')] + sys.path)
        gpus = [ i for (i, _) in resources['gpus'] ]
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpus)
        env['NV_GPU'] = ','.join(str(g) for g in gpus)
        container_id = None
        if self.parameters:
            with open(os.path.join(self._dir, self.PARAMS), 'w') as f:
                f.write(yaml.dump(dict(self.parameters)))
        user_args = copy.copy(self.user_args)
        user_uid = pwd.getpwnam(self.username).pw_uid
        # prepare docker job parameters
        job_real_dir = get_real_path(self._dir)
        args = ['/usr/bin/docker', 'create', '--runtime=nvidia', '--rm']
        args.extend(['-e', 'NVIDIA_VISIBLE_DEVICES='+env['NV_GPU']])
        args.extend(['-e', 'JOB_DIR=/workspace'])
        metrics_path = os.path.join("/workspace", self.METRICS)
        args.extend(['-e', 'METRICS_PATH={}'.format(metrics_path)])
        #args.extend(['-u', '{}:{}'.format(user_uid, user_uid)])
        if self.dataset_path:
            dataset_host_path = get_real_path(self.dataset_path)
            args.extend(['-v', dataset_host_path+':/dataset'])
        # check for user arguments
        for n, arg in enumerate(user_args):
            if "JOB_DIR" in arg:
                user_args[n] = arg.replace("JOB_DIR", "/workspace")
            if self.dataset_path:
                if "DATASET_DIR" in arg:
                    user_args[n] = arg.replace("DATASET_DIR", "/dataset")
        with open(os.path.join(self._dir, "user_args.sh"), 'w') as f:
            f.write(' '.join(user_args))
        args.extend(['-v', job_real_dir+':/workspace','-w', self._workspace, self.image_tag, "bash", "-x", "/workspace/user_args.sh"])
        run_args = ' '.join(args)
        logger.info("Run {}".format(run_args))
        try:
            output = subprocess.check_output(run_args, stderr=subprocess.STDOUT,
                                             shell=True, universal_newlines=True)
        except subprocess.CalledProcessError as exc:
            for line in exc.output.split('\n'):
                self.output_log.write('%s\n' % line)
            self.after_run()
            raise exc
        logger.info("Run output: {}".format(output))
        container_id = re.findall(r"^\w+", output)[0][:6]
        args = ["/usr/bin/docker", "start", "-i", container_id]
        run_args = ' '.join(args)
        # End of docker start
        logger.info("Run {}".format(run_args))
        self.p = subprocess.Popen(run_args,
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  cwd=self._dir,
                                  close_fds=True,
                                  env=env,
                                  )
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
                        subprocess.check_output("/usr/bin/docker ps |grep {}".format(container_id), shell=True)
                    except:
                        ## Bug, docker start may hang while container not exist.
                        break
                        #self.abort()
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
 

def _repo_name(repo_path):
    find = re.match(".*/(.*).git", repo_path)
    if find:
        return find.group(1)
    find = re.match(".*/(.*)", repo_path)
    if find:
        return find.group(1)
    raise ValueError("Can't find repo name for {}".format(repo_path))


def pkg_clone(repo_path, parent_dir, child_dir):
    repo_name = _repo_name(repo_path)
    src_to = os.path.join(child_dir, repo_name)
    #if not os.path.exists(src_to):
    #    os.makedirs(src_to)
    src_from = os.path.join(parent_dir, repo_name)
    logger.debug("Copy pkg {} to {}".format(src_from, src_to))
    system_copy(src_from, src_to)


def new(username, name, image_tag, dataset_path, user_args,
        num_gpu, project, repo_path, parameters, parent):
    while True:
        id = RUNS_PREFIX + JOBS_PREFIX + str(uuid.uuid4()).replace('-', '')[:8]
        job_dir = os.path.join(RUNS_DIR, id)
        if not os.path.exists(job_dir):
            break
    try:
        args = user_args.split()
        for n, arg in enumerate(args):
            if arg.startswith("bk/"):
                args[n] = arg.replace('bk/', '/data/dataset/')
        if dataset_path:
            if dataset_path.startswith("bk/"):
                dataset_path = dataset_path.replace('bk/', '/data/dataset/')
            if not os.path.exists(dataset_path):
                raise ValueError("Cannot find dataset {}".format(dataset_path))
        inst = Instance(id=id,
                        username=username,
                        name=name,
                        image_tag=image_tag,
                        dataset_path=dataset_path,
                        user_args=args,
                        num_gpu=num_gpu,
                        project=project,
                        status_history=[],
                        repo_path=repo_path,
                        parameters=parameters,
                        parent=parent,
                        child=[])
        logger.debug("Create instance {}".format(inst.id))
    except Exception as e:
        logger.warning('Caught %s while creating instance %s: %s' % (type(e).__name__, id, e))
        logger.debug(traceback.format_exc(e))
        raise e
    return inst


def params_parser(params, p_list):
    has_list = False
    for k, v in params.iteritems():
        if type(v) == list:
            has_list = True
            for sv in v:
                c_params = params.copy()
                c_params[k] = sv
                params_parser(c_params, p_list)
            return
    if has_list == False:
        p_list.append(params)


def create(username, job_name, image_tag, dataset_path, user_args,
           num_gpu, project, repo_path, parameters, parent, replicate=False):
    params_list = []
    inst_list = []
    params_parser(parameters, params_list)
    if replicate:
        # skip code path if copy from job
        repo_path = ''
    for params in params_list:
        instance = new(username, job_name, image_tag, dataset_path,
                       user_args, num_gpu, project, repo_path, params, parent)
        inst_list.append(instance)
    if len(inst_list) > 1:
        for inst in inst_list[1:]:
            inst_list[0].child.append(inst.id)
            inst.parent = inst_list[0].id
    for inst in inst_list:
        inst.status = INIT
        inst.save()
    # only return first instance
    return inst_list[0].id


#def commit(username, inst):
#    if not inst.module_name:
#        return
#    # create and commit to gogs
#    repo_name = inst.module_name.split('.')[0] + "-" + inst.id[-3:]
#    gogsop.repo_create(username, repo_name)
#    pkg_path = os.path.join(inst._dir, inst.module_name.split('.')[0])
#    gogsop.repo_commit(username, pkg_path, repo_name)
#    inst.sha = gogsop.commit_last(username, repo_name)
#    inst.repo_path = "ssh://root@gogs:222/{}/{}.git".format(username, repo_name)
#    inst.save()


def repo_clone(username, inst, repo_path):
    gogsop.repo_clone(username, repo_path, inst._dir)
    inst.repo_path = repo_path
    inst.save()


def delete_by_id(job_id):
    inst = Instance.load(job_id)
    logger.debug("delete instance {}".format(job_id))
    job_dir = os.path.join(RUNS_DIR, job_id)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)


def system_copy(src, dst):
    check_output("cp -PRT {} {}".format(src, dst), shell=True)


def rsync_copy(src, dst):
    check_output("rsync -a --exclude='info.json' --exclude='output.log' --exclude='param.yml' --exclude='metrics.csv' {} {}".format(src, dst), shell=True)
