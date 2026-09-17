from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import Settings
from app.core.migrations import sync_sqlite_columns

_engine: Engine | None = None


def init_engine(settings: Settings) -> Engine:
    global _engine
    url = settings.sqlalchemy_url
    is_sqlite = url.startswith("sqlite")
    engine = create_engine(url, connect_args={"check_same_thread": False} if is_sqlite else {})
    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    import app.models.entities  # noqa: F401  (register tables)

    SQLModel.metadata.create_all(engine)
    sync_sqlite_columns(engine)
    if _engine is not None:
        _engine.dispose()
    _engine = engine
    return engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Database engine not initialised")
    return _engine


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
