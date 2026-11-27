"""Persistent SQLAlchemy 2 storage layer for Astropilote."""

from astropilote.db.models import (
    Base,
    CatalogProgress,
    ChatMessage,
    ChatThread,
    MagicLinkToken,
    ObservingSession,
    PendingActionApproval,
    Rig,
    SessionTarget,
    Site,
    User,
    UserApiKey,
)
from astropilote.db.session import (
    LOCAL_USER_ID,
    create_db_engine,
    db_session,
    ensure_local_user,
    get_db,
    get_engine,
    init_db,
)

__all__ = [
    "LOCAL_USER_ID",
    "Base",
    "CatalogProgress",
    "ChatMessage",
    "ChatThread",
    "MagicLinkToken",
    "ObservingSession",
    "PendingActionApproval",
    "Rig",
    "SessionTarget",
    "Site",
    "User",
    "UserApiKey",
    "create_db_engine",
    "db_session",
    "ensure_local_user",
    "get_db",
    "get_engine",
    "init_db",
]
