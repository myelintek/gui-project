from __future__ import absolute_import

import os
import tempfile

from . import option_list


if 'JOBS_DIR' in os.environ:
    value = os.environ['JOBS_DIR']
else:
    value = os.path.join('/data', 'jobs')


try:
    value = os.path.abspath(value)
    if os.path.exists(value):
        if not os.path.isdir(value):
            raise IOError('No such directory: "%s"' % value)
        if not os.access(value, os.W_OK):
            raise IOError('Permission denied: "%s"' % value)
    if not os.path.exists(value):
        os.makedirs(value)
    projects_path = os.path.abspath(os.path.join(value, 'projects'))
    if not os.path.exists(projects_path):
        os.makedirs(projects_path)
except:
    print '"%s" is not a valid value for jobs_dir.' % value
    print 'Set the envvar JOBS_DIR to fix your configuration.'
    raise


option_list['jobs_dir'] = value
