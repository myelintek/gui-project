import click
from myelindl.client import MyelindlApi, MyelindlApiError


@click.command()
def list():
    try:
        api = MyelindlApi()
        result = api.image_list()
        longest = 20
        if result:
            longest = max(len(x['tag']) for x in result)
        if longest < 20:
            longest = 20
        template = '| {:>10} | {:>%d} |'% longest
        header = template.format('id', 'tag')
        click.echo('=' * len(header))
        click.echo(header)
        click.echo('=' * len(header))
        for image in result:
            line = template.format(image['id'], image['tag'])
            click.echo(line)
        click.echo('=' * len(header))
    except MyelindlApiError as e:
        click.echo(str(e))


@click.command()
@click.option('--id', required=True, help="image id")
def delete(id):
    try:
        api = MyelindlApi()
        result = api.image_delete(id)
        click.echo(result) 
    except MyelindlApiError as e:
        click.echo(str(e))


@click.command()
@click.option('--tag', required=True, help="image tag")
def pull(tag):
    try:
        api = MyelindlApi()
        result = api.image_pull(tag)
        click.echo(result) 
    except MyelindlApiError as e:
        click.echo(str(e))


@click.group(help='Group of commands to manage container images')
def image():
    pass

image.add_command(list)
image.add_command(delete)
image.add_command(pull)


@click.group(help='Groups of commands to manage container')
def container():
    pass

container.add_command(image)
