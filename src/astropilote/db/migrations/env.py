"""Alembic migration environment supporting both local SQLite WAL and cloud PostgreSQL."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from astropilote.config import settings
from astropilote.db.models import Base

config = context.config
target_metadata = Base.metadata


def _get_url() -> str:
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured
    return settings.resolved_database_url()


def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    cfg_section = config.get_section(config.config_ini_section) or {}
    cfg_section["sqlalchemy.url"] = _get_url()
    connectable = engine_from_config(
        cfg_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
