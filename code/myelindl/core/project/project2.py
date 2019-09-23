import os
import json
import time
import uuid
from threading import Lock

from datetime import datetime

from ..dataset import bucket
from ..models import (
    db,
    User,
    Project,
    ProjectMembers,
    ProjectMemberPermission,
)

DEFAULT_PATH = '/data/dataset'
PROJECT_METADATA_PATH = '/data/dataset/projects.json'
lock = Lock()


class Project2(object):
    def __init__(self, id, username, name, dataset, is_member_only=False, notes=''):
        self.id = id
        self.username = username
        self.name = name
        self.create_time = time.time()
        self.notes = notes
        self.dataset = dataset
        self.jobs = 0
        self.info_path = PROJECT_METADATA_PATH
        self.clone_path = "http://localhost:3000/default/"+name+".git"
        self.is_member_only = is_member_only

    def save_to_db(self):
        p_db = Project()
        p_db.legacy_id = self.id
        p_db.owner = self.username
        p_db.is_member_only = self.is_member_only

        u = User.query.filter(User.username == self.username).first()
        if u:
            m = ProjectMembers()
            m.user = u
            m.permission = ProjectMemberPermission.OWNER
            p_db.members.append(m)
        db.session.add(p_db)
        db.session.commit()

    @property
    def _dict(self):
        d = self.__dict__.copy()
        return d

    @property
    def time(self):
        strtime = datetime.fromtimestamp(self.create_time).strftime("%Y %b %d, %H:%M:%S")
        return strtime


def projects_load():
    try:
        with open(PROJECT_METADATA_PATH, 'r') as infile:
            data = json.load(infile)
    except Exception as e:
        return []
    return data


def projects_list(username=None):
    projects = projects_load()
    # TODO later for multi-tenant
    # if username not in ['', None]:
    #     projects = list(filter(lambda d : d['username'] == username, projects))
    if not username:
        return projects
    results = [p for p in projects if has_permission(p['id'], username, ProjectMemberPermission.PERMISSIONS)]
    return results


def projects_save(projects):
    if not os.path.exists(DEFAULT_PATH):
        os.makedirs(DEFAULT_PATH)
    with open(PROJECT_METADATA_PATH, 'w') as outfile:
        json.dump(projects, outfile)


def project_delete(username, _id):
    with lock:
        projects = projects_load()
        project = None
        for p in projects:
            if p['id'] == _id:
                project = p
                break

        if project is None:
            raise ValueError('Project {} not found!'.format(_id))

        p = Project.query.filter(Project.legacy_id == _id).first()
        if p:
            for m in p.members:
                db.session.delete(m)
            db.session.delete(p)
            db.session.commit()

        projects.remove(project)
        projects_save(projects)


def project_get(name):
    projects = projects_load()
    for p in projects:
        if p['name'] == name:
            return p
    return None


def project_get_id(id):
    projects = projects_load()
    for p in projects:
        if p['id'] == id:
            return p
    return None


def project_get_name_from_id(id):
    projects = projects_load()
    for p in projects:
        if p['id'] == id:
            return p['name']
    return None


def project_new(username, name, dataset, is_member_only=False,  notes=''):
    if project_get(name) is not None:
        raise ValueError('Project {} exist!'.format(name))
    bucket.check_exist(dataset)
    with lock:
        projects = projects_load()
        while True:
            _id = str(uuid.uuid4()).replace('-', '')[:8]
            if _id not in projects:
                break
        project = Project2(_id, username, name, dataset, is_member_only=is_member_only, notes=notes)
        # add old jobs to same project name
        from myelindl.webapp import scheduler
        insts = scheduler.get_instances()
        count = 0
        for inst in insts:
            if project.name == inst.project:
                count += 1
        project.jobs = count
        project.save_to_db()
        projects.append(project._dict)
        projects_save(projects)

    return project


def project_update(id, **kwargs):
    with lock:
        projects = projects_load()
        project = None
        for p in projects:
            if p['id'] == id:
                project = p

        if not project:
            raise ValueError('Project {} not found'.format(id))

        for k, v in kwargs.items():
            project[k] = v
        project_model_update(id, **kwargs)
        projects_save(projects)


def project_job_update(id):
    from myelindl.webapp import scheduler
    p = project_get_id(id)
    insts = scheduler.get_instances()
    count = 0
    for inst in insts:
        if inst.project == p['name']:
            count += 1
    project_update(p['id'], jobs=count)


def project_model_update(id, **kwargs):
    project = Project.query.filter(Project.legacy_id == id).first()
    if not project:
        raise ValueError('Project {} not found'.format(id))

    try:
        for k, v in kwargs.items():
            if hasattr(project, k):
                setattr(project, k, v)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise e


def add_member(id, username, permission):
    user = User.query.filter(User.username == username).first()
    if not user:
        raise ValueError('User {} not found'.format(username))

    project = Project.query.filter(Project.legacy_id == id).first()
    if not project:
        raise ValueError('Project {} not found'.format(id))

    member = ProjectMembers.query.\
        filter(ProjectMembers.project_id == project.id).\
        filter(ProjectMembers.user_id == user.id).\
        first()

    if not member:
        member = ProjectMembers()
        member.user = user
        member.permission = permission
        project.members.append(member)
        db.session.add(project)
    else:
        if project.owner == username:
            raise ValueError('Can not change owners permission')
        member.permission = permission

    db.session.commit()


def delete_member(id, username):
    user = User.query.filter(User.username == username).first()
    if not user:
        raise ValueError('User {} not found'.format(username))
    project = Project.query.filter(Project.legacy_id == id).first()
    if not project:
        raise ValueError('Project {} not found'.format(id))

    if project.owner == username:
        raise ValueError('Can not delete owner')

    member = ProjectMembers.query.\
        filter(ProjectMembers.project_id == project.id).\
        filter(ProjectMembers.user_id == user.id).\
        first()

    if not member:
        raise ValueError('Proejct {} Member {} not found.'.format(id, username))
    db.session.delete(member)
    db.session.commit()


def is_legal_permission(permission):
    return permission in ProjectMemberPermission.PERMISSIONS


def has_permission(id, username, permissions):
    project = Project.query.filter(Project.legacy_id == id).first()
    if not project:
        raise ValueError('Project {} not found'.format(id))

    if not project.is_member_only:
        return True

    if project.owner == username:
        return True

    user = User.query.filter(User.username == username).first()
    if not user:
        raise ValueError('User {} not found'.format(username))

    project_member = ProjectMembers.query.\
        filter(ProjectMembers.project_id == project.id).\
        filter(ProjectMembers.user_id == user.id).\
        first()

    if not project_member:
        return False

    if project_member.permission not in permissions:
        return False

    return True


def list_member(id):
    members = []
    project = Project.query.filter(Project.legacy_id == id).first()
    if not project:
        return members

    for m in project.members:
        members.append({
            'name': m.user.username,
            'permission': m.permission,
        })

    return members
