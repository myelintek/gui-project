import os
import click
from .frameworks import FrameworkManager
from myelindl.core.utils import uff_converter
from myelindl.client import MyelindlApi, MyelindlApiError
import getpass
import model_lib
from luminoth.utils.config import get_config, override_config_params
from luminoth.utils.config import load_config_files


SUPPORT_CONVERT_FORMATS = ['uff']
framework = FrameworkManager()

@click.command()
@click.option('--model', type=click.Choice(framework.support_models()), required=True)
@click.option('--force', is_flag=True, default=False)
def download(model, force):
    framework.download(model, force)


@click.command()
@click.option('--model', type=click.Choice(framework.support_models()), required=True)
@click.option('--input_dir', required=True, help='Where to load the input files')
@click.option('--score_threshold', required=False, type=click.FLOAT, help='Minimum box score.')
def inference(model, input_dir, score_threshold):
    framework.inference(model, input_dir, score_threshold)


@click.command()
@click.option('--model', type=click.Choice(framework.support_models()), required=True)
def server(model):
    try:
        api = MyelindlApi()
        result = api.create_server(model)
        server_id = result['server_id']
        print ("Server runing on  http://localhost/models/server/{}/view".format(server_id))
    except MyelindlApiError as e:
        click.echo(str(e))


@click.command()
@click.option('--model', type=click.Choice(framework.support_models()), required=True)
@click.option('--format', type=click.Choice(SUPPORT_CONVERT_FORMATS), required=True)
@click.option('--output_dir', required=True)
def convert(model, format, output_dir):
    sess, graph, output_dict = framework.get_model_convert_output(model)
    uff_converter.convert_uff_from_tensorflow(sess, graph, output_dict, output_dir)


@click.command()
def list():
    checkpoints = None 
    try:
        checkpoints = framework.list_models()
    except Exception as e:
        click.echo(str(e))
        return

    template = '| {:>12} | {:>21} | {:>11} | {:>6} | {:>14} |'

    header = template.format(
        'id', 'name', 'alias', 'source', 'status'
    )
    click.echo('=' * len(header))
    click.echo(header)
    click.echo('=' * len(header))

    for checkpoint in checkpoints :
        line = template.format(
            checkpoint['id'],
            checkpoint['name'],
            checkpoint['alias'],
            checkpoint['source'],
            checkpoint['status'],
        )
        click.echo(line)

    click.echo('=' * len(header))      
    

@click.command(help='Create a checkpoint from a configuration file.')
@click.argument('config_files', nargs=-1)
@click.option(
    'override_params', '--override', '-o', multiple=True,
    help='Override model config params.'
)
@click.option(
    'entries', '--entry', '-e', multiple=True,
    help="Specify checkpoint's metadata field value."
)
def create(config_files, override_params, entries):
    new_config = load_config_files(config_files)
    if new_config.framework == 'pytorch':
        model_lib.create4PyTorch(new_config, override_params=override_params, entries=entries)
    else:
        from luminoth.tools.checkpoint import create as lumi_create
        lumi_create(config_files, override_params, entries)


@click.group(help='Gropup of command to manage models')
def model():
    pass


#from luminoth.tools.checkpoint import create
from luminoth.tools.checkpoint import delete

model.add_command(inference)
model.add_command(download)
model.add_command(list)
model.add_command(server)
model.add_command(convert)
model.add_command(create)
model.add_command(delete)

