from __future__ import absolute_import

import os
import re

from . import option_list
import myelindl


value = os.path.join('/code', 'aug', 'image')

try:
    aug_list = []
    value = os.path.abspath(value)
    if os.path.exists(value):
        if not os.path.isdir(value):
            raise IOError('No such directory: "%s"' % value)
        if not os.access(value, os.W_OK):
            raise IOError('Permission denied: "%s"' % value)
    if not os.path.exists(value):
        raise IOError('No such directory: "%s"' % value)
    for filename in os.listdir(value):
        path = os.path.join(value, filename)
        match = None
        match = re.match(r'^[a-z].*.py$', filename)
        if match:
            content = None
            with open(path) as infile:
                content = infile.read()
            aug_list.append((filename, content))
except:
    print '"%s" is not a valid value for aug_dir'
    raise

option_list['aug_list'] = aug_list
