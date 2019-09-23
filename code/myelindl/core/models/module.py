from . import db


class Module(db.Model):
    __tablename__ = 'module'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128, collation='NOCASE'), nullable=False, unique=True)
    version = db.Column(db.String(128, collation='NOCASE'), nullable=True)

