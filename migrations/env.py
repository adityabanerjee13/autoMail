"""Alembic environment.

Runs against the same psycopg3 DSN the application uses, in blocking mode --
migrations are the one place where async buys nothing.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from triage.config import get_settings
from triage.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
settings = get_settings()


def _url() -> str:
    """An explicit -x url / sqlalchemy.url wins over the application settings.

    This used to read ``settings.database_url`` unconditionally, which meant a
    caller that passed a different URL was silently ignored and the migration
    ran against the live database instead. That is a quiet way to lose data: a
    migration test pointed at a scratch database ran ``downgrade base`` against
    production and dropped every table. Honouring the override is what makes
    "run this against another database" mean what it says.
    """
    override = config.get_main_option("sqlalchemy.url", None)
    return override or settings.database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
