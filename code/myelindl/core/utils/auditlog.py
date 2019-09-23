from __future__ import absolute_import

import re
import sys
import time
import logging
from datetime import datetime

from myelindl.core.models import db, AuditLog
from myelindl.log import DATE_FORMAT

PROJECT = 'Project'
DATASET = 'Dataset'
USER = 'User'
SERVICE = 'Service' 
CONTAINER = 'Container'
MODEL = 'Model'
SYSTEM = 'System'

DECODE_RE = r"AUDIT username:(.*), type:(.*), msg:(.*)$"


def decode_msg(encoded_msg):
    """ Decode encoded log string
    """
    match = re.match(DECODE_RE, encoded_msg)
    
    if not match:
        return None

    return {
        'username': match.group(1),
        'type': match.group(2),
        'message': match.group(3),
    }


def encode_msg(username, type, message):
    """ Encode log string
    """
    return "AUDIT username:{}, type:{}, msg:{}".format(
        username,
        type,
        message
    )


class AuditlogDBHandler(logging.Handler):

    def __init__(self, db):
        super(AuditlogDBHandler, self).__init__()
        self.db = db
    
    def emit(self, record):
        decoded = decode_msg(record.msg)
        
        if not decoded:
            return

        audit_log = AuditLog()
        audit_log.time = datetime.fromtimestamp(record.created)
        audit_log.level = record.levelno
        audit_log.username = decoded['username']
        audit_log.type = decoded['type']
        audit_log.message = decoded['message']
        
        try:
            self.db.session.add(audit_log)
            self.db.session.commit()
        except Exception as e:
            self.db.session.rollback()
            raise


class AuditlogFilter(logging.Filter):
    def filter(self, record):
        return record.getMessage().startswith('AUDIT')

formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt=DATE_FORMAT,
)

auditlog_handler = AuditlogDBHandler(db)
auditlog_handler.addFilter(AuditlogFilter())
auditlog_handler.setLevel(logging.INFO)
auditlog_handler.setFormatter(formatter)

audit_logger = logging.getLogger('myelindl.core.utils.auditlog')
audit_logger.addHandler(auditlog_handler)


def log(level, username, type, message, **kwargs):
    """ Log Audit log 
    """
    msg = encode_msg(username, type, message)
    audit_logger.log(level, msg, **kwargs)


def critical(username, type, message, **kwargs):
    log(logging.CRITICAL, username, type, message, **kwargs)


def error(username, type, message, **kwargs):
    log(logging.ERROR, username, type, message, **kwargs)


def warning(username, type, message, **kwargs):
    log(logging.WARNING, username, type, message, **kwargs)


def info(username, type, message, **kwargs):
    log(logging.INFO, username, type, message, **kwargs)


def debug(username, type, message, **kwargs):
    log(logging.DEBUG, username, type, message, **kwargs)


def list_log(username=None, type=None, message=None, level=None, 
        time_start=None, time_end=None, offset=0, limit=50):
    """ Search Related log
    """
    filters = {}
    query = AuditLog.query

    if username:
        query = query.filter(AuditLog.username == username)
    
    if type:
        query = query.filter(AuditLog.type == type)

    if message:
        query = query.filter(AuditLog.message.like('%{}%'.format(message)))

    if level:
        query = query.filter(AuditLog.level == level)

    if time_start:
        query = query.filter(AuditLog.time >= time_start)

    if time_end:
        query = query.filter(AuditLog.time <= time_end)
    
    query = query.offset(offset).limit(limit).orderby(AuditLog.time.desc())
    
    return query.all()
