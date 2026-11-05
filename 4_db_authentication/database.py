"""JSON database interface for face authentication."""

from pathlib import Path

from face_core import FaceStore
from models import ROOT


DEFAULT_DATABASE_PATH = ROOT / "data" / "embeddings.json"


def initialize_database(path=DEFAULT_DATABASE_PATH):
    """Initialize and return the JSON-based face store."""
    return FaceStore(path)


def add_user(store, name, samples):
    """Register a user using face embedding samples."""
    store.enroll(name, samples)


def delete_user(store, name):
    """Delete a registered user."""
    store.delete(name)


def get_users(store):
    """Return registered user names."""
    return list(store.people.keys())


def get_database_path():
    """Return the path of the JSON database."""
    return Path(DEFAULT_DATABASE_PATH)
