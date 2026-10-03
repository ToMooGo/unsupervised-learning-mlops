"""Database schema shared by the API (writes) and the pipeline / monitoring flows (reads).

PostgreSQL in Docker Compose; any SQLAlchemy URL works (SQLite is used in tests).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One row per /predict request: input, both classifiers' outputs and the anomaly verdict."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    source: Mapped[str] = mapped_column(String(32), default="api")
    pixels: Mapped[list] = mapped_column(JSON)
    supervised_digit: Mapped[int] = mapped_column(Integer)
    supervised_confidence: Mapped[float] = mapped_column(Float)
    semi_supervised_digit: Mapped[int] = mapped_column(Integer)
    semi_supervised_confidence: Mapped[float] = mapped_column(Float)
    log_density: Mapped[float] = mapped_column(Float)
    density_threshold: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, index=True)
    model_versions: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)


class HumanLabel(Base):
    """Labels typed by a person on the web UI's labelling page (one per representative image)."""

    __tablename__ = "human_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    split_seed: Mapped[int] = mapped_column(Integer, index=True)
    train_index: Mapped[int] = mapped_column(Integer, index=True)
    label: Mapped[int] = mapped_column(Integer)
    labeler: Mapped[str] = mapped_column(String(64), default="anonymous")


def make_engine(url: str) -> Engine:
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_db(engine: Engine) -> sessionmaker:
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def fetch_human_labels(url: str, split_seed: int) -> dict[int, int]:
    """Latest human label per training index for the given split."""
    engine = make_engine(url)
    Session = init_db(engine)
    with Session() as s:
        rows = s.execute(
            select(HumanLabel)
            .where(HumanLabel.split_seed == split_seed)
            .order_by(HumanLabel.created_at)
        ).scalars()
        labels = {r.train_index: r.label for r in rows}
    engine.dispose()
    return labels
