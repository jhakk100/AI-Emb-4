"""Face authentication interface using the team's JSON FaceStore."""

from face_core import DEFAULT_THRESHOLD


def authenticate_face(store, embedding, threshold=DEFAULT_THRESHOLD):
    """
    Authenticate a face embedding against registered users.

    Returns:
        name: matched user name or None
        score: highest similarity score or None
    """
    return store.match(embedding, threshold)


def check_threshold(score, threshold):
    """Check whether a similarity score passes the threshold."""
    return score >= threshold
