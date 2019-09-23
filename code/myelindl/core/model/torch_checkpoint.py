
import click
import json
import os
import shutil
import six
import requests
import uuid
from datetime import datetime
import errno
import db_lib

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

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

def download_torch_remote_checkpoint(db, checkpoint):
    # Check if output directory doesn't exist already to fail early.
    import getpass
    output = os.path.join(db_lib.get_model_path(getpass.getuser()), checkpoint['id'])
    if os.path.exists(output):
        click.echo(
            "Checkpoint directory '{}' for checkpoint_id '{}' already exists. "
            "Try issuing a `lumi checkpoint delete` or delete the directory "
            "manually.".format(output, checkpoint['id'])
        )
        return

    # build output path
    makedirs(output)

    # Start the pkl file download.
    pkl_fname = checkpoint['url'].split('/')[-1]
    response = requests.get(checkpoint['url'], stream=True)
    length = int(response.headers.get('Content-Length'))
    chunk_size = 16 * 1024
    progressbar = click.progressbar(
        response.iter_content(chunk_size=chunk_size),
        length=length / chunk_size, label='Downloading checkpoint...',
    )
    with open(os.path.join(output, pkl_fname), 'wb') as f:
        with progressbar as content:
            for chunk in content:
                f.write(chunk)

    # Start the npy file download.
    npy_fname = checkpoint['url2'].split('/')[-1]
    response = requests.get(checkpoint['url2'], stream=True)
    length = int(response.headers.get('Content-Length'))
    chunk_size = 16 * 1024
    progressbar = click.progressbar(
        response.iter_content(chunk_size=chunk_size),
        length=length / chunk_size, label='Downloading checkpoint...',
    )
    with open(os.path.join(output, npy_fname), 'wb') as f:
        with progressbar as content:
            for chunk in content:
                f.write(chunk)

    # Import the checkpoint from the tar.
    click.echo("Importing checkpoint... ", nl=False)
    click.echo("done.")

    # Update the checkpoint status and persist database.
    checkpoint['status'] = 'DOWNLOADED'
    save_checkpoint_db(db)

    click.echo("Checkpoint imported successfully.")

