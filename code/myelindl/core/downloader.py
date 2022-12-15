import os
import uuid
import requests
import tempfile
import tarfile
from myelindl.log import logger
from myelindl.env import RUNS_DIR, RUNS_PREFIX
from myelindl.core.run import INIT, WAIT, RUN, DONE, ABORT, ERROR, Run
from myelindl.webapp import scheduler


class Downloader(Run):
    def __init__(self, url, path, username):
        id = RUNS_PREFIX + str(uuid.uuid4()).replace('-', '')[:8]
        super(Downloader, self).__init__(id, username, 'downloader',
              status_history=[], persistent=False)
        self.url = url
        self.download_path = path
        scheduler.add_instance(self)

    def run(self, resources):
        self.before_run()
        tempdir = tempfile.mkdtemp()
        tmpfile = os.path.join(tempdir, 'temp.tar')
        logger.info("Downloading {}... ".format(self.url))
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            length = int(r.headers.get('Content-Length'))
            dl = .0
            with open(tmpfile, 'wb') as f:
                for chunk in r.iter_content(chunk_size=16*1024):
                    dl += len(chunk)
                    f.write(chunk)
                    self.progress = dl/length
        logger.info("Importing {}... ".format(self.download_path))
        with tarfile.open(tmpfile) as f:
            members = [m for m in f.getmembers()]
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.download_path, members)
        logger.info("done.")

        self.after_run()
        self.status = DONE
