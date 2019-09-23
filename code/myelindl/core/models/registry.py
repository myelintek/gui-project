from . import db

class ContainerRegistryCredential(db.Model):
    __tablename__ = 'container_registry_creds'

    id = db.Column(db.Integer, primary_key=True)
    registry_name = db.Column(db.String(255), nullable=False)
    credentials = db.Column(db.String())
