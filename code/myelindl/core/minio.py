import os
import re
import uuid
import json
import logging
from subprocess import check_output


logger = logging.getLogger('myelindel.core.minio')

KEYS_PATH = "/data/dataset/keys.json"


def key_gen(username):
    logger.debug("username: {}".format(username))
    keys_data = {'keys': []}
    if os.path.exists(KEYS_PATH):
        with open(KEYS_PATH, 'r') as f:
            keys_data = json.load(f)
        logger.debug("keys_data: {}".format(keys_data))
        for key in keys_data['keys']:
            if key['username'] == username:
                logger.debug(key['secretkey'])
                # ensure minio has this user registered
                os.system("mc admin user add mlsteam {} {} readwrite".format(username, key['secretkey']))
                return key['secretkey']
    # Create new user
    secret = str(uuid.uuid4()).replace('-', '')[:10]
    keys_data['keys'].append({'username': username, 'secretkey': secret})
    with open(KEYS_PATH, 'w') as f:
        f.write(json.dumps(keys_data, sort_keys=True, indent=4))
    os.system("mc admin user add mlsteam {} {} readwrite".format(username, secret))
    return secret


def bucket_delete(name):
    try:
        check_output("mc rb --force mlsteam/{}".format(name), shell=True)
    except Exception as e:
        logger.warning("remove bucket {} failed. {}".format(name, str(e)))


def bucket_list():
    try:
        output = check_output("mc ls mlsteam/", shell=True)
        buckets = []
        for line in output.splitlines():
            match = re.match('\[(.*)\]\s+(\d+B) (\w+)', line)
            if match:
                bk_time = match.group(1)
                bk_name = match.group(3)
                buckets.append({'name':bk_name, 'time':bk_time})
        return buckets
    except Exception as e:
        return []
