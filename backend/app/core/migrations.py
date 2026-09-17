"""Minimal additive schema sync for SQLite.

`create_all` never alters existing tables. Until Alembic is justified (PostgreSQL / destructive
changes), new *nullable or defaulted* columns are added with ALTER TABLE ADD COLUMN.
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel

logger = logging.getLogger("voicelab.migrations")


def sync_sqlite_columns(engine: Engine) -> list[str]:
    if engine.dialect.name != "sqlite":
        return []
    inspector = inspect(engine)
    added: list[str] = []
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                if not column.nullable and column.default is None and column.server_default is None:
                    logger.error("migration_needed", extra={"table": table.name, "column": column.name})
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                default = ""
                if column.default is not None and getattr(column.default, "is_scalar", False):
                    value = column.default.arg
                    default = f" DEFAULT {int(value) if isinstance(value, bool) else repr(value)}"
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}{default}'))
                added.append(f"{table.name}.{column.name}")
    if added:
        logger.info("schema_columns_added", extra={"columns": added})
    return added
