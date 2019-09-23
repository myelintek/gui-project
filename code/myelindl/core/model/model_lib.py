import click
import db_lib
import shutil
from luminoth.utils.config import load_config_files
import glob, os
import uuid
from datetime import datetime
import getpass


def create4PyTorch(new_config, override_params=None, entries=None):
    # Retrieve the related files for inferencing later.    
    run_dir = os.path.join(new_config.train.job_dir, "*.pkl")
    files = glob.glob(run_dir)
    files.sort(key=os.path.getmtime)
    pkl_file = files[-1] # get the latest one
    npy_file = glob.glob(os.path.join(new_config.train.package_path, "*.npy"))[0]
    
    checkpoint_id = str(uuid.uuid4()).replace('-', '')[:12]
    new_config.dataset.dir = '.'
    new_config.train.job_dir = '.'
    new_config.train.run_name = checkpoint_id

    # Create checkpoint path
    path = db_lib.get_model_path(getpass.getuser(), checkpoint_id)
    os.makedirs(path, 0755)
    
    # Add the pickle and npy file
    shutil.copy2(pkl_file, path)
    shutil.copy2(npy_file, path)

    # Store the new checkpoint into the checkpoint index.
    metadata = {
        'id': checkpoint_id,
        'name': new_config.train.run_name,
        'description': '',
        'alias': new_config.alias,
    
        'model': new_config.model,
        'dataset': {
            'dir': path,
            'name': new_config.dataset.name,
            'num_classes': new_config.dataset.num_classes
        },
        'framework': new_config.framework, 
        'luminoth_version': lumi_version,
        'created_at': datetime.utcnow().isoformat(),
    
        'status': 'LOCAL',
        'source': 'local',
        'url': None 
    }

    db = db_lib.read_model_db()
    db['checkpoints'].append(metadata)
    save_checkpoint_db(db)

    click.echo('Checkpoint {} created successfully.'.format(checkpoint_id))


