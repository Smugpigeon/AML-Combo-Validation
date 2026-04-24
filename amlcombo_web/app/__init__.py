"""amlcombo_web — public-facing web service for the AML Combo-Prediction Kit.

Thin web wrapper around the `combo_val` package:
  - FastAPI REST + Jinja2 HTML pages
  - PostgreSQL for users/jobs/reports metadata
  - Redis + Celery for background kit execution
  - Per-user API keys (ac_live_...) for programmatic access
  - BYOK: users store encrypted OpenAI/Anthropic keys for optional LLM
    report enhancement (server never sees plaintext except transit)

Deployed as a single-VPS Docker Compose stack behind Cloudflare + Caddy.

Target users: AML researchers (self-service). Regulatory posture:
research use only, per-report disclaimer. No PHI — de-identified inputs
only.
"""
