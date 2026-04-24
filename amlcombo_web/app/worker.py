"""Celery app instance — imported by both the web app (to enqueue) and
by the worker process (to consume)."""

from __future__ import annotations

from celery import Celery

from app.config import get_settings


_settings = get_settings()

celery_app = Celery(
    "amlcombo",
    broker=_settings.CELERY_BROKER_URL,
    backend=_settings.CELERY_RESULT_BACKEND,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_max_tasks_per_child=50,  # recycle workers to limit RAM leaks
    task_time_limit=10 * 60,         # 10-minute hard limit per task
    task_soft_time_limit=8 * 60,
)
