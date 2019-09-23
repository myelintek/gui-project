
import os
import json

from luminoth.tools.checkpoint import (
    get_checkpoint,
    download_remote_checkpoint,
    read_checkpoint_db,
    get_checkpoints_directory,
    save_checkpoint_db,
    CHECKPOINT_INDEX,
    CHECKPOINT_PATH,
    merge_index,
    get_config,
    get_checkpoint_config,
)
from easydict import EasyDict as edict
import getpass

def get_model_path(username, cpt_id=None):
    if 'root' == username:
        base_path = '/root/.luminoth'
    else:
        base_path = '/home/%s/.luminoth' % username

    if cpt_id:
        return os.path.join(base_path, CHECKPOINT_PATH, cpt_id)
    else:
        return os.path.join(base_path, CHECKPOINT_PATH)


def read_model_db(username=None):
    if not username:
        username = getpass.getuser()
    path = os.path.join(get_model_path(username), CHECKPOINT_INDEX)
    if not os.path.exists(path):
        return {'checkpoints': []}
    
    with open(path) as f:
        index = json.load(f) 
    return index


def get_model_config(username, network, prompt=False):
    db = read_model_db(username)
    checkpoint = get_checkpoint(db, network)
    if not checkpoint:
        raise ValueError('i Checkpoint not found.')
    if checkpoint['status'] == 'NOT_DOWNLOADED':
        raise ValueError('i Checkpoint not downloaded.')

    path = get_model_path(username, checkpoint['id'])

    config_path = os.path.join(path, 'config.yml')
    if not os.path.exists(config_path):
        return edict({
            'model': {'type': checkpoint['model']},
            'dataset': {'dir': path },
            'train': {'job_dir': get_model_path(username)}
        })
    
    try: 
        config = get_config(config_path)
    except Exception:
        return edict({
            'model': {'type': checkpoint['model']},
            'dataset': {'dir': path },
            'train': {'job_dir': get_model_path(username)}
        })

    # Config paths should point to the path where the checkpoint files are
    # stored.
    config.dataset.dir = path
    config.train.job_dir = get_model_path(username)
    return config


def ensure_default_db():
    path = os.path.join(get_checkpoints_directory(), CHECKPOINT_INDEX)
    if not os.path.exists(path):
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_config_path = os.path.join(base_dir, 'config', CHECKPOINT_INDEX)
        with open(default_config_path) as f:
            default_db = json.load(f)
        db = read_checkpoint_db()
        db = merge_index(db, default_db)
        save_checkpoint_db(db)


#
# make function alias
#
download_remote_model = download_remote_checkpoint
get_model = get_checkpoint

#
# initial db file
#
ensure_default_db()

