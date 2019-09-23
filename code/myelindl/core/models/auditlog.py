from . import db

class AuditLog(db.Model):
    __tablename__ = 'auditlog'

    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, nullable=False)
    level = db.Column(db.String(8), nullable=False)
    username = db.Column(db.String(128), nullable=True)
    type = db.Column(db.String(64), nullable=False)
    message = db.Column(db.String(256), nullable=False)

