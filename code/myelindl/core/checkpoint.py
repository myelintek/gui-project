import os
import json
import uuid
import shutil
import requests
import tempfile
from myelindl.log import logger
from myelindl.core.downloader import Downloader
from myelindl.env import RUNS_DIR, RUNS_PREFIX
from myelindl.webapp import scheduler


CHECKPOINT_INDEX = 'checkpoints.json'
CHECKPOINT_PATH = '/data/checkpoints'
REMOTE_INDEX_URL = (
    'https://github.com/tryolabs/luminoth/releases/download/v0.0.3/'
    'checkpoints.json'
)


def ensure_default_db():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config_path = os.path.join(base_dir, 'config', CHECKPOINT_INDEX)
    with open(default_config_path) as f:
        default_db = json.load(f)
    
    db = {'checkpoints':[]}
    db = merge_index(db, default_db)
#    url = REMOTE_INDEX_URL
#    logger.info('Retrieving remote index... ')
#    remote = requests.get(url).json()
#    logger.info('done.')
#    db = merge_index(db, remote)
    save_checkpoint_db(db)
    return db


# checkpoint list
def read_checkpoint_db():
    """Reads the checkpoints database file from disk."""
    path = os.path.join(CHECKPOINT_PATH, CHECKPOINT_INDEX)
    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)
    if not os.path.exists(path):
        ensure_default_db()
        if not os.path.exists(path):
            return {'checkpoints': []}

    with open(path) as f:
        index = json.load(f)

    return index


# checkpoint download
def download(id_or_alias, username):
    logger.info("download checkpoint {}".format(id_or_alias))
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, id_or_alias)
    if not checkpoint:
        raise ValueError(
            "Checkpoint '{}' not found in index.".format(id_or_alias)
        )

    if checkpoint['source'] != 'remote':
        raise ValueError(
            "Checkpoint is not remote. If you intended to download a remote "
            "checkpoint and used an alias, try using the id directly."
        )

    if checkpoint['status'] != 'NOT_DOWNLOADED':
        raise ValueError("Checkpoint is already downloaded.")
    logger.info("checkpoint: {}".format(checkpoint))
    download_remote_checkpoint(checkpoint, username)


class CPDownloader(Downloader):
    def __init__(self, url, path, id, username):
        super(CPDownloader, self).__init__(url, path, username)
        self.checkpoint_id = id

    def after_run(self):
        self.output_log.close()
        self.progress = 1.
        # Update the checkpoint status and persist database.
        checkpoint_set_download(self.checkpoint_id)


def checkpoint_set_download(id):
    db = read_checkpoint_db()
    checkpoint = get_checkpoint(db, id)
    if checkpoint:
        checkpoint['status'] = 'DOWNLOADED'
    save_checkpoint_db(db)


def download_remote_checkpoint(checkpoint, username):
    # Check if output directory doesn't exist already to fail early.
    output = get_checkpoint_path(checkpoint['id'])
    if os.path.exists(output):
        shutil.rmtree(output)
        logger.info("delete existing checkpoint {}.".format(output))
    logger.info("create CPDownloader")
    CPDownloader(checkpoint['url'], output, checkpoint['id'], username)


def get_checkpoint_path(checkpoint_id):
    """Returns checkpoint's directory."""
    return os.path.join(CHECKPOINT_PATH, checkpoint_id)


def get_checkpoint(db, id_or_alias):
    """Returns checkpoint entry in `db` indicated by `id_or_alias`.

    First tries to match an ID, then an alias. For the case of repeated
    aliases, will first match local checkpoints and then remotes. In both
    cases, matching will be newest first.
    """
    # Go through the checkpoints ordered by creation date. There shouldn't be
    # repeated aliases, but if there are, prioritize the newest one.
    local_checkpoints = sorted(
        [c for c in db['checkpoints'] if c['source'] == 'local'],
        key=lambda c: c['created_at'], reverse=True
    )
    remote_checkpoints = sorted(
        [c for c in db['checkpoints'] if c['source'] == 'remote'],
        key=lambda c: c['created_at'], reverse=True
    )

    selected = []
    for cp in local_checkpoints:
        if cp['id'] == id_or_alias or cp['alias'] == id_or_alias:
            selected.append(cp)

    for cp in remote_checkpoints:
        if cp['id'] == id_or_alias or cp['alias'] == id_or_alias:
            selected.append(cp)

    if len(selected) < 1:
        return None

    if len(selected) > 1:
        logger.info(
            "Multiple checkpoints found for '{}' ({}). Returning '{}'.".format(
                id_or_alias, len(selected), selected[0]['id']
            )
        )

    return selected[0]




def merge_index(local_index, remote_index):
    """Merge the `remote_index` into `local_index`.

    The merging process is only applied over the checkpoints in `local_index`
    marked as ``remote``.
    """

    non_remotes_in_local = [
        c for c in local_index['checkpoints']
        if c['source'] != 'remote'
    ]
    remotes_in_local = {
        c['id']: c for c in local_index['checkpoints']
        if c['source'] == 'remote'
    }

    to_add = []
    seen_ids = set()
    for checkpoint in remote_index['checkpoints']:
        seen_ids.add(checkpoint['id'])
        local = remotes_in_local.get(checkpoint['id'])
        if local:
            # Checkpoint is in local index. Overwrite all the fields.
            local.update(**checkpoint)
        elif not local:
            # Checkpoint not found, it's an addition. Transform into our schema
            # before appending to `to_add`. (The remote index schema is exactly
            # the same except for the ``source`` and ``status`` keys.)
            checkpoint['source'] = 'remote'
            checkpoint['status'] = 'NOT_DOWNLOADED'
            to_add.append(checkpoint)

    # Out of the removed checkpoints, only keep those with status
    # ``DOWNLOADED`` and turn them into local checkpoints.
    missing_ids = set(remotes_in_local.keys()) - seen_ids
    already_downloaded = [
        c for c in remotes_in_local.values()
        if c['id'] in missing_ids and c['status'] == 'DOWNLOADED'
    ]
    for checkpoint in already_downloaded:
        checkpoint['status'] = 'LOCAL'
        checkpoint['source'] = 'local'

    new_remotes = [
        c for c in remotes_in_local.values()
        if not c['id'] in missing_ids  # Checkpoints to remove.
    ] + to_add + already_downloaded

    if len(to_add):
        logger.info('{} new remote checkpoints added.'.format(len(to_add)))
    if len(missing_ids):
        if len(already_downloaded):
            logger.info('{} remote checkpoints transformed to local.'.format(
                len(already_downloaded)
            ))
        logger.info('{} remote checkpoints removed.'.format(
            len(missing_ids) - len(already_downloaded)
        ))
    if not len(to_add) and not len(missing_ids):
        logger.info('No changes in remote index.')

    local_index['checkpoints'] = non_remotes_in_local + new_remotes

    return local_index


def save_checkpoint_db(checkpoints):
    """Overwrites the database file in disk with `checkpoints`."""
    path = os.path.join(CHECKPOINT_PATH, CHECKPOINT_INDEX)
    with open(path, 'w') as f:
        json.dump(checkpoints, f)
