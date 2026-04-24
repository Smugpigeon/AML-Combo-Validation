"""Celery tasks — background jobs triggered from the web layer."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.db import db_session
from app.kit_runner import run_kit_prediction
from app.models import Submission, UsageEvent
from app.worker import celery_app

log = logging.getLogger(__name__)


@celery_app.task(name="app.tasks.run_patient_kit", bind=True, max_retries=1)
def run_patient_kit(self, submission_id: str) -> dict:
    """Run the kit on a queued Submission and persist results.

    Updates the Submission row through its status machine:
      queued → running → done | failed
    """
    with db_session() as db:
        sub = db.get(Submission, submission_id)
        if sub is None:
            log.warning("Submission %s not found", submission_id)
            return {"error": "not_found"}
        sub.status = "running"
        sub.started_at = datetime.now(tz=timezone.utc)
        db.flush()

        user_id = str(sub.user_id)
        input_json = dict(sub.input_json or {})
        rna_counts_path = sub.rna_counts_path
        rna_full_path = sub.rna_full_path

    try:
        result = run_kit_prediction(
            user_id=user_id,
            submission_id=str(submission_id),
            input_json=input_json,
            rna_counts_csv=rna_counts_path,
            rna_full_csv=rna_full_path,
        )
    except Exception as e:  # pylint: disable=broad-except
        log.exception("Kit run failed for submission %s", submission_id)
        with db_session() as db:
            sub = db.get(Submission, submission_id)
            if sub is not None:
                sub.status = "failed"
                sub.error_message = str(e)[:1000]
                sub.finished_at = datetime.now(tz=timezone.utc)
        return {"status": "failed", "error": str(e)[:500]}

    # Persist result
    with db_session() as db:
        sub = db.get(Submission, submission_id)
        if sub is None:
            return {"error": "not_found_post_run"}
        sub.status = "done"
        sub.finished_at = datetime.now(tz=timezone.utc)
        sub.report_md_path = result.get("report_md_path")
        sub.report_pdf_path = result.get("report_pdf_path")
        sub.report_html_path = result.get("report_html_path")
        sub.report_figure_path = result.get("report_figure_path")
        sub.dna_summary_json = result.get("dna_summary")
        sub.rna_outlier_json = result.get("rna_outlier")
        sub.predicted_eln2017 = result.get("predicted_eln2017")
        sub.top_regimen_name = result.get("top_regimen_name")
        # Audit event
        db.add(UsageEvent(
            user_id=sub.user_id,
            event_type="kit_run_completed",
            meta={"submission_id": str(submission_id),
                  "eln": result.get("predicted_eln2017")},
        ))
    return {"status": "done", "submission_id": str(submission_id)}
