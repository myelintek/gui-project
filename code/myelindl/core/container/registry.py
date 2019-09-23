import json

import docker

from ..models import db, ContainerRegistryCredential

class Registry(object):
    def login (self, docker):
        ''' Extend this to user docker client login'''
        pass

class NGCRegistry(Registry):
    def __init__(self, **kwargs):
        self.api_key = kwargs.pop('api_key')
 
    def login(self, docker):
        docker.login(username='$oauthtoken', password=self.api_key, registry='nvcr.io')

REGISTRIES = {
    'nvcr.io': NGCRegistry,
}

REGISTRY_CRED_FIELDS = {
    'nvcr.io': [('api_key', True)],
}

def list_registry():
    return REGISTRIES.keys()


def get_registry_cred_fields(registry_name):
    return REGISTRY_CRED_FIELDS[registry_name]


def test_credential(registry_name, **kwargs):
    cli = None
    try:
        cli = docker.APIClient(base_url='unix://var/run/docker.sock')
        registry = get_registry(registry_name, kwargs)
        registry.login(cli)
    finally:        
        if cli:
            cli.close()

def get_registry(registry_name, cred=None):
    ''' Registry factory method
        get registry instance by name, if registry didnot have credential
        return Registery (do nothing whent login)
    '''
    if registry_name in [None, '']:
        return Registry()
    if not cred:
       cred = get_credential(registry_name)
    if not cred:
        return Registry()
    
    return REGISTRIES[registry_name](**cred)


def add_credential(registry_name, **kwargs):
    ''' store credential of specified registry_name'''
    registry = ContainerRegistryCredential()
    registry.registry_name = registry_name
    registry.credentials = json.dumps(kwargs)
    try:
        db.session.add(registry)
        db.session.commit()
    except:
        db.session.rollback()
        raise

def list_credentials():
    return ContainerRegistryCredential.query.all()


def get_credential(registry_name):
    registry = ContainerRegistryCredential.query.filter(
        ContainerRegistryCredential.registry_name == registry_name,
    ).first()
    
    if not registry:
        return None
    
    return json.loads(registry.credentials)

def delete_credential(registry_name):
    ''' Delete credential of specified registry_name'''
    registry = ContainerRegistryCredential.query.filter(
        ContainerRegistryCredential.registry_name == registry_name,
    ).first()

    if not registry:
        raise ValueError('Registry {} not found'.format(registry_name))
    try:
        db.session.delete(registry)
        db.session.commit()
    except:
        db.session.rollback()
        raise

