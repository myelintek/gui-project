from . import db


class ProjectType(object):
    GUI = 'GUI'
    IDE = 'IDE'


class ProjectPermission(object):
    PUBLIC = 'PUBLIC'
    MEMBER_ONLY = 'MEMBER_ONLY'


class ProjectMemberPermission(object):
    '''
              delete proj  adduser   train   view 
    Owner :        v          v        v       v
    Writer:        x          x        v       v
    Reader:        x          x        x       v
    '''

    OWNER = 'Owner'
    WRITER = 'Writer'
    READER = 'Reader'

    PERMISSIONS = [OWNER, WRITER, READER]

    PERM_WRITE = [OWNER, WRITER]
    PERM_READ = PERMISSIONS
    PERM_DELETE = [OWNER]
    PERM_MEMBER = [OWNER]


class Project(db.Model):
    __tablename__ = 'projects'

    id = db.Column(db.Integer, primary_key=True)
    legacy_id = db.Column(db.String(128), nullable=False, unique=True)
    is_member_only = db.Column(db.Boolean, unique=False, default=False)
    owner = db.Column(db.String(128), unique=False, default=False)
    members = db.relationship('ProjectMembers')


class ProjectMembers(db.Model):
    __tablename__ = 'project_members'

    project_id = db.Column(db.Integer, db.ForeignKey('projects.id', ondelete='CASCADE'), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    permission = db.Column(db.String(64, collation='NOCASE'), nullable=False)
    user = db.relationship('User')
