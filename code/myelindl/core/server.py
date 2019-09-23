
from myelindl.core.run import INIT, WAIT, RUN, DONE, ABORT, ERROR, Run
from myelindl.env import RUNS_DIR, RUNS_PREFIX

SERVER_PREFIX = 'server-'


class Server(Run):
    def __init__(self, id, username, name, checkpoint,
                 status_history=[]):
        super(Server, self).__init__(id, username, name, status_history)
        self.checkpoint = checkpoint

    def offer_resources(self, resources):
        if 'gpus' not in resources:
            return None
        for resource in resources['gpus']:
            if resource.remaining() >= 1:
                return {'gpus': [(resource.identifier, 1)]}
        return {}  # don't use a GPU at all

    def run(self, resources):
        self.before_run()
        env = os.environ.copy()
        gpus = [ i for (i, _) in resources['gpus'] ]
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpus)
        env['NV_GPU'] = ','.join(str(g) for g in gpus)
 
