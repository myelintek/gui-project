import os
import uuid
import click
import tensorflow as tf
from ftplib import FTP
from myelindl.core.util import sizeof_fmt
from myelindl.client import MyelindlApi, MyelindlApiError
from .readers import get_reader, READERS
from .writers import get_writer

@click.command()
@click.option('dataset_type', '--type', type=click.Choice(READERS.keys()), required=True)  # noqa
@click.option('--data-dir', required=True, help='Where to locate the original data.')  # noqa
@click.option('--output-dir', required=True, help='Where to save the transformed data.')  # noqa
@click.option('splits', '--split', required=True, multiple=True, help='The splits to transform (ie. train, test, val).')  # noqa
@click.option('--debug', is_flag=True, help='Set level logging to DEBUG.')
def conversion(dataset_type, data_dir,  output_dir, splits, debug):
    tf.logging.set_verbosity(tf.logging.INFO)
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)

    try:
        reader = get_reader(dataset_type)
    except ValueError as e:
        tf.logging.error('Error getting reader: {}'.format(e))
        return

    try:
        writer = get_writer(dataset_type)
        for split in splits:
            split_reader = reader(data_dir, split)
            writer = writer(split_reader, output_dir, split)
            writer.save()

            tf.logging.info('Composition per class ({})'.format(split))
    except ValueError as e:
        tf.logging.error('Error reading dataset: {}'.format(e))


@click.command()
@click.option('name', '--name', required=True, help='Dataset name')
@click.option('--data-dir', required=True, help="Where to locate the directory contains coverted data.")
@click.option('--id', required=False, help="Specify id to update published dataset.")
@click.option('--type', required=False, help="Specify type of published dataset.(ex: object, file)")
@click.option('--description', required=False, help='Dataset description')
def publish(data_dir, name, id, type, description):

    if not os.path.exists(data_dir):
        click.echo('data-dir: {} not exists!'.format(data_dir))

    temp_dir = 'mlt_tmp-{}'.format(str(uuid.uuid1())[:8])
    try:
        api = MyelindlApi()

        api.data_upload(data_dir, temp_dir)

        dataset_id = id if id else ''
        dataset_type = type if type else 'file'
        
        result = api.publish_dataset(
            id=dataset_id,
            type=dataset_type,
            name=name,
            description=description,
            data_dir=temp_dir,
        )
        
        click.echo('Dataset {} published with id {}'.format(name, result['id']))
    except MyelindlApiError as e:
        click.echo(u'Dataset publish fail, due to {}'.format(str(e)))


@click.command()
@click.option('id', '--id', required=True, help="Which dataset you want to unpublish")
def unpublish(id):
    api = MyelindlApi()
    try: 
        result = api.unpublish_dataset(id)
        click.echo('Dataset {} unpublished'.format(id))
    except MyelindlApiError as e :
        click.echo('Fail due to {}'.format(str(e)))
        return


@click.command()
def list():
    api = MyelindlApi()
    try: 
        result = api.list_publish_dataset()
    except MyelindlApiError as e :
        click.echo('Fail due to {}'.format(str(e)))
        return
    
    template = '| {:>10} | {:>10} | {:>30} | {:>10} | {:>10} | {:>10} |'
    header = template.format('id', 'name', 'description','size', 'type', 'user')

    click.echo('=' * len(header))
    click.echo(header)
    click.echo('=' * len(header))

    for ds in result['datasets']:
        line = template.format(
            ds['id'],
            ds['name'],
            ds['description'],
            sizeof_fmt(ds['size']),
            ds['type'],
            ds['username'],
        )
        click.echo(line)
    click.echo('='* len(header))


@click.command()
@click.option('id', '--id', required=True, help="Which dataset you want to read information")
def info(id):
    api = MyelindlApi()
    try: 
        result = api.get_dataset_info(id)
    except MyelindlApiError as e :
        click.echo('Fail due to {}'.format(str(e)))
        return
    
    template = '| {:>20} | {:>30}|'
    header = template.format('Field', 'Value')

    click.echo('=' * len(header))
    click.echo(header)
    click.echo('=' * len(header))

    for k, v in result.iteritems():
        line = template.format(k, v)
        click.echo(line)
    click.echo('='* len(header))


@click.command()
@click.option('--id',required=True)
@click.option('dir','--dir',required=False)
def browse(id, dir):

    api = MyelindlApi()
    try: 
        result = api.get_dataset_items(id,dir)
        if 'files' not in result:
            raise Exception('Browse unsupported dataset type.'
                    'We only support browse file type of dataset')

        for file in result['files']:
            if file['is_dir']:
                click.echo('{}/'.format(file['name']))
            else:
                click.echo(file['name'])
    except Exception as e:
        click.echo("Fail due to {}".format(str(e)))
@click.group(help='Groups of commands to manage datasets')
def dataset():
    pass


dataset.add_command(conversion)
dataset.add_command(publish)
dataset.add_command(unpublish)
dataset.add_command(list)
dataset.add_command(browse)
dataset.add_command(info)
