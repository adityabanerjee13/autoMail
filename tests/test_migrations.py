"""The migrations actually run, in both directions.

test_queue.py builds its schema from ``Base.metadata`` rather than from Alembic,
on the stated grounds that "a broken migration shows up in a migration test,
not here". This is that test, which until now did not exist -- so nothing
checked that the ORM models and the migrations agreed.

    docker compose up -d postgres
    TEST_DATABASE_URL=postgresql+psycopg://triage:triage@127.0.0.1:5432/triage_test \
        pytest tests/test_migrations.py
"""

from __future__ import annotations

import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from triage.config import REPO_ROOT

TEST_DSN = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TEST_DSN, reason="set TEST_DATABASE_URL to run migration tests"),
]

CORE_TABLES = {"messages", "classifications", "jobs", "sync_state"}
CHAT_TABLES = {"chat_threads", "chat_messages"}


def _refuse_to_touch_the_real_database() -> None:
    """These tests run `downgrade base`. Never against the working database.

    env.py used to ignore the URL passed to it and use the application settings
    instead, so pointing this suite at a scratch database silently dropped
    every table in the real one. env.py is fixed, but this belt-and-braces
    check stays: the failure is total data loss, and it is completely silent
    until you go looking for your mail.
    """
    from triage.config import get_settings

    live = get_settings().database_url
    if TEST_DSN == live:
        pytest.fail(
            "TEST_DATABASE_URL is the application's own database. These tests "
            "drop every table. Point it at a scratch database instead."
        )


@pytest.fixture
def alembic_cfg():
    _refuse_to_touch_the_real_database()
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DSN)
    return cfg


@pytest.fixture
def engine():
    # Sync driver: alembic runs its migrations synchronously.
    eng = create_engine(TEST_DSN.replace("+psycopg", "+psycopg"), future=True)
    yield eng
    eng.dispose()


def table_names(engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def test_upgrade_head_creates_every_table(alembic_cfg, engine):
    command.downgrade(alembic_cfg, "base")
    command.upgrade(alembic_cfg, "head")
    assert CORE_TABLES | CHAT_TABLES <= table_names(engine)


def test_downgrade_to_0001_removes_only_the_chat_tables(alembic_cfg, engine):
    """0002 is additive; rolling it back must not disturb the mail schema."""
    command.upgrade(alembic_cfg, "head")
    command.downgrade(alembic_cfg, "0001")

    names = table_names(engine)
    assert not (CHAT_TABLES & names), "chat tables survived the downgrade"
    assert CORE_TABLES <= names, "downgrading 0002 damaged the mail schema"

    command.upgrade(alembic_cfg, "head")


def test_the_migration_matches_the_orm_models(alembic_cfg, engine):
    """The models are the autogenerate target; they must describe what 0002 built.

    Catches the classic drift where a column is added to models.py and the
    migration is forgotten -- which would pass every unit test, because
    test_queue.py builds its schema from the models themselves.
    """
    from triage.db.models import ChatMessageRow, ChatThreadRow

    command.upgrade(alembic_cfg, "head")
    insp = inspect(engine)
    for model in (ChatThreadRow, ChatMessageRow):
        actual = {c["name"] for c in insp.get_columns(model.__tablename__)}
        expected = {c.name for c in model.__table__.columns}
        assert expected == actual, f"{model.__tablename__} drifted: {expected ^ actual}"


def test_chat_messages_keeps_its_constraints(alembic_cfg, engine):
    """The append-only trigger is deliberately absent here; the CHECKs are not."""
    command.upgrade(alembic_cfg, "head")
    insp = inspect(engine)
    checks = {c["name"] for c in insp.get_check_constraints("chat_messages")}
    assert {"ck_chat_messages_role", "ck_chat_messages_status"} <= checks

    indexes = {i["name"] for i in insp.get_indexes("chat_messages")}
    assert "ix_chat_messages_live" in indexes
