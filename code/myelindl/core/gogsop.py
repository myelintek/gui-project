import os
import re
import json
import requests
import myelindl
import subprocess
from myelindl.core.util import copytree
import logging

logger = logging.getLogger(__name__)

TOKEN_PATH="/data/.gogs/auth_token.txt"
GOGS_ADMIN="root"
GOGS_PASS="P@ssw0rd"


def token_create(username, password):
    url = "http://localhost:3000/api/v1/users/{}/tokens".format(username)
    response = requests.post(url,
                             auth=(username, password),
                             json={"name": username})
    if response.status_code < 300 and response.status_code >= 200:
        token_file = "/data/.gogs/{}_token.txt".format(username)
        with open(token_file, "w") as f:
            f.write(response.text)
        logger.debug("Got {} token {}".format(username, response.text))


def token_get(username):
    token_file = "/data/.gogs/{}_token.txt".format(username)
    with open(token_file, "r") as f:
        token_str = f.read()
        token_dict = json.loads(token_str)
        return token_dict['sha1']


def repo_create(username, repo_name):
    token = token_get(username)
    url = "http://localhost:3000/api/v1/user/repos?token={}".format(token)
    res = requests.post(
        url=url,
        json={"name": repo_name, "private": False},
    )
    try:
        res.raise_for_status()
    except Exception as e:
        raise ValueError('Repository already exist, {}'.format(repo_name))


def _repo_name(repo_path):
    find = re.match(".*/(.*).git", repo_path)
    if find:
        return find.group(1)
    find = re.match(".*/(.*)", repo_path)
    if find:
        return find.group(1)
    raise ValueError("Can't find repo name for {}".format(repo_path))


def repo_clone(username, repo_path, job_dir):
    # sync sshkey & clone
    cmd = os.path.join(os.path.dirname(os.path.abspath(myelindl.__file__)),
                       'tools', 'gogs_clone.py')
    command = "su -c 'python {} {}' - {}".format(cmd, repo_path, username)
    logger.info("{}".format(command))
    subprocess.call(command, shell=True)
    repo_name = _repo_name(repo_path)
    # copy code to job folder
    src_from = "/home/{}/.mlsteam/{}".format(username, repo_name)
    src_to = os.path.join(job_dir, repo_name)
    if not os.path.exists(src_to):
        os.makedirs(src_to)
    logger.info("copy repository {} to {}".format(src_from, src_to))
    copytree(src_from, src_to)


def repo_delete(username, repo_path):
    repo_name = _repo_name(repo_path)
    token = token_get(username)
    url = "http://localhost:3000/api/v1/repos/{}/{}?token={}".format(
        username, repo_name, token)
    res = requests.delete(url=url)
    res.raise_for_status()


def repo_commit(username, package_path, repo_name):
    cmd = os.path.join(os.path.dirname(os.path.abspath(myelindl.__file__)),
                       'tools', 'initial_commit.py')
    command = "su -c 'python {} {} {}' - {}".format(
        cmd, repo_name, package_path, username)
    logger.debug("{}".format(command))
    subprocess.call(command, shell=True)


def key_get(username):
    token = token_get(username)
    url = "http://localhost:3000/api/v1/user/keys?token={}".format(token)
    res = requests.get(url)
    res_json = json.loads(res.text)
    for key in res_json:
        if key['title'] == "autogen":
            return key['key']
    return ''


def key_remove(username):
    token = token_get(username)
    url = "http://localhost:3000/api/v1/user/keys?token={}".format(token)
    res = requests.get(url)
    res_json = json.loads(res.text)
    for key in res_json:
        if key['title'] == "autogen":
            url = "http://localhost:3000/api/v1/user/keys/{}?token={}".format(key['id'], token)
            res = requests.delete(url)
            res.raise_for_status()


def key_update(username, update_key):
    key = key_get(username)
    if key:
        # always remove existing key
        logger.debug("Remove existing key")
        key_remove(username)
    logger.debug("Adding key: {}".format(update_key))
    token = token_get(username)
    url = "http://localhost:3000/api/v1/user/keys?token={}".format(token)
    res = requests.post(
        url=url,
        json={"title": "autogen", "key": update_key},
    )
    res.raise_for_status()


def commit_last(username, repo_name):
    token = token_get(username)
    url = "http://localhost:3000/api/v1/repos/{}/{}/commits/master?token={}".format(
        username, repo_name, token)
    try:
        res = requests.get(
            url=url,
        )
        res.raise_for_status()
    except Exception as e:
        return ''
    info = json.loads(res.text)
    return info['sha'][:10]
