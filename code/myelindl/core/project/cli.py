import os
import json
import click
import getpass
import requests
#from . import project2
from datetime import datetime
#from myelindl.utils import gogs
from myelindl.client import MyelindlApi, MyelindlApiError
from shutil import copy
PROJECT_METADATA_PATH = '/data/jobs/projects.json'


@click.command()
@click.option('name', '--name', required=True, help='project name')
def create(name):
    try:
        api = MyelindlApi()
        result = api.project_create(name)

        #click.echo('{}'.format(project_id))
    except MyelindlApiError, e:
        click.echo("new project failed, {}".format(e))


@click.command()
@click.option('--json', 'is_json', default=False, help="return json format output")
def list(is_json):
    try:
        api = MyelindlApi()
        result = api.project_list()

        if is_json:
            click.echo(json.dumps(result, indent=2, sort_keys=True))
            return
        template = '| {:>8} | {:>10} | {:>8} | {:>20} |'
        header = template.format('id', 'name', 'user', 'create time')
        click.echo('=' * len(header))
        click.echo(header)
        click.echo('=' * len(header))
        for project in result:
            line = template.format(project['id'],
                                   project['name'],
                                   project['username'],
                                   datetime.fromtimestamp(project['create_time']).strftime("%Y %b %d, %H:%M:%S"))
            click.echo(line)
        click.echo('=' * len(header))
    except MyelindlApiError, e:
        click.echo("list project failed, {}".format(e))


@click.command()
@click.option('--id', required=True, help="id of the project")
def delete(id):
    try:
        api = MyelindlApi()
        name = api.project_get_info(id)['info']['name']
        result = api.project_delete(id)
    except MyelindlApiError, e:
        click.echo("delete a project failed, {}".format(e))


@click.command()
@click.option('name', '--name', required=True, help='project name')
def pull(name):
    
    ret=os.system("git clone http://localhost:3000/default/"+name+".git")
    if ret != 0:
        raise ValueError('cannot pull repo '+str(name))
    copy("/build/build_conf/pre-push",os.getcwd()+"/"+name+"/.git/hooks/")
    os.chmod(os.getcwd()+"/"+name+"/.git/hooks/pre-push", 0770)
    

@click.command()
@click.option('id', '--id', required=True, help='id')
def info(id):
    try:
        api = MyelindlApi()
        result=api.project_get_info(id)

        template = '| {:>11} | {:>40}|'
        header = template.format('Field', 'Value')
        click.echo('=' * len(header))
        click.echo(header)
        click.echo('=' * len(header))

        for k, v in result['info'].iteritems():
            line = template.format(k, v)
            click.echo(line)
        click.echo('='* len(header))
    except MyelindlApiError, e:
        click.echo("show project info failed, {}".format(e))


@click.group(help='Groups of commands to manage project')
def project():
    pass

project.add_command(create)
project.add_command(list)
project.add_command(delete)
project.add_command(pull)
project.add_command(info)
