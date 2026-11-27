"""Initial Astropilote v1.2 schema (users, BYOK keys, rigs, sites, observing logbook, chat, approvals).

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=True, unique=True),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("provider_subject", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("provider", "provider_subject", name="uq_users_provider_subject"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "user_api_keys",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("encrypted_key", sa.Text(), nullable=False),
        sa.Column("key_hint", sa.String(length=32), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "provider", name="uq_user_api_keys_provider"),
    )
    op.create_index("ix_user_api_keys_user_id", "user_api_keys", ["user_id"])

    op.create_table(
        "rigs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("preset_id", sa.String(length=64), nullable=True),
        sa.Column("focal_length_mm", sa.Float(), nullable=False),
        sa.Column("aperture_mm", sa.Float(), nullable=True),
        sa.Column("sensor_format", sa.String(length=48), nullable=False),
        sa.Column("sensor_width_mm", sa.Float(), nullable=False),
        sa.Column("sensor_height_mm", sa.Float(), nullable=False),
        sa.Column("mount_type", sa.String(length=32), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_rigs_user_id", "rigs", ["user_id"])

    op.create_table(
        "sites",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("bortle_class", sa.Integer(), nullable=True),
        sa.Column("elevation_m", sa.Float(), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_sites_user_id", "sites", ["user_id"])

    op.create_table(
        "observing_sessions",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("obs_date", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column(
            "site_id",
            sa.String(length=64),
            sa.ForeignKey("sites.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("site_name", sa.String(length=120), nullable=False),
        sa.Column("lat", sa.Float(), nullable=True),
        sa.Column("lon", sa.Float(), nullable=True),
        sa.Column(
            "rig_id",
            sa.String(length=64),
            sa.ForeignKey("rigs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("rig_name", sa.String(length=120), nullable=False),
        sa.Column("cloud_cover_pct", sa.Float(), nullable=True),
        sa.Column("moon_illumination_pct", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_observing_sessions_user_id", "observing_sessions", ["user_id"])
    op.create_index("ix_observing_sessions_obs_date", "observing_sessions", ["obs_date"])

    op.create_table(
        "session_targets",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=64),
            sa.ForeignKey("observing_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("object_id", sa.String(length=32), nullable=False),
        sa.Column("object_name", sa.String(length=120), nullable=False),
        sa.Column("ra_deg", sa.Float(), nullable=True),
        sa.Column("dec_deg", sa.Float(), nullable=True),
        sa.Column("exposure_minutes", sa.Float(), nullable=False),
        sa.Column("sub_exposure_s", sa.Float(), nullable=True),
        sa.Column("frames_count", sa.Integer(), nullable=False),
        sa.Column("filter_name", sa.String(length=48), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False),
    )
    op.create_index("ix_session_targets_session_id", "session_targets", ["session_id"])
    op.create_index("ix_session_targets_object_id", "session_targets", ["object_id"])

    op.create_table(
        "catalog_progress",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("object_id", sa.String(length=32), nullable=False),
        sa.Column("catalog", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("first_observed_at", sa.String(length=16), nullable=True),
        sa.Column("total_exposure_minutes", sa.Float(), nullable=False),
        sa.Column(
            "best_session_id",
            sa.String(length=64),
            sa.ForeignKey("observing_sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.UniqueConstraint("user_id", "object_id", name="uq_catalog_progress_user_obj"),
    )
    op.create_index("ix_catalog_progress_user_id", "catalog_progress", ["user_id"])
    op.create_index("ix_catalog_progress_object_id", "catalog_progress", ["object_id"])

    op.create_table(
        "chat_threads",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_threads_user_id", "chat_threads", ["user_id"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "thread_id",
            sa.String(length=64),
            sa.ForeignKey("chat_threads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chat_messages_thread_id", "chat_messages", ["thread_id"])

    op.create_table(
        "magic_link_tokens",
        sa.Column("token_hash", sa.String(length=64), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_magic_link_tokens_email", "magic_link_tokens", ["email"])

    op.create_table(
        "pending_action_approvals",
        sa.Column("token_hash", sa.String(length=64), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_pending_action_approvals_user_id", "pending_action_approvals", ["user_id"])


def downgrade() -> None:
    op.drop_table("pending_action_approvals")
    op.drop_table("magic_link_tokens")
    op.drop_table("chat_messages")
    op.drop_table("chat_threads")
    op.drop_table("catalog_progress")
    op.drop_table("session_targets")
    op.drop_table("observing_sessions")
    op.drop_table("sites")
    op.drop_table("rigs")
    op.drop_table("user_api_keys")
    op.drop_table("users")
