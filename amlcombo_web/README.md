# amlcombo_web

Web service for the AML Combo-Prediction Kit — the FastAPI layer at `amlcombo.org`.

Thin wrapper around the `combo_val` package:
- REST API + Jinja2 HTML UI
- Per-user API keys for programmatic access
- BYOK: users store their own encrypted OpenAI/Anthropic keys (LLM usage billed to them)
- Celery background worker runs the kit (~30-120s/patient)
- PostgreSQL + Redis on a single VPS via Docker Compose

## Quick start — local dev

```bash
cd amlcombo_web
docker compose -f docker-compose.dev.yml up  # or just pip install + python -m uvicorn
```

Or without Docker:
```bash
pip install -e ..                  # install combo_val as editable
pip install -e .[dev]
# Generate dev secrets:
export JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export DATABASE_URL=sqlite:///./dev.db
export STORAGE_ROOT=./dev_storage
export KIT_ASSETS_ROOT=../data/canonical
export APP_ENV=dev
uvicorn app.main:app --reload
```

Visit http://localhost:8000.

## Run tests

```bash
cd amlcombo_web
python -m pytest tests/ -v
# 17 tests, <3 seconds
```

## Deploy to production

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full Hetzner + Cloudflare + Namecheap walkthrough.

## Architecture

```
Browser → Cloudflare → Caddy → FastAPI (web) ─┐
                                              ├→ Postgres
                                              ├→ Redis
                                              └→ Celery worker → runs combo_val kit
```

## Key files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app entrypoint |
| `app/config.py` | All env var configuration |
| `app/models.py` | SQLAlchemy ORM (User, APIKey, LLMKey, Submission) |
| `app/security.py` | bcrypt + JWT + API key minting |
| `app/crypto.py` | Fernet encryption for user LLM keys |
| `app/deps.py` | FastAPI auth dependencies (cookie + API key) |
| `app/llm.py` | OpenAI/Anthropic BYOK client |
| `app/kit_runner.py` | combo_val kit adapter |
| `app/tasks.py` | Celery background tasks |
| `app/worker.py` | Celery app instance |
| `app/routers/` | REST endpoints |
| `app/templates/` | Jinja2 HTML |
| `docker-compose.yml` | 5-service prod stack |
| `Dockerfile` | Multi-purpose image (web + worker) |
| `Caddyfile` | Reverse proxy + Let's Encrypt |
| `docs/DEPLOYMENT.md` | VPS rental + DNS + deploy guide |

## Security notes

- All passwords bcrypt-hashed
- All API keys sha256-hashed (plaintext shown ONCE at creation)
- All user LLM keys Fernet-encrypted at rest (AES-128-CBC + HMAC-SHA256)
- Session cookies HTTP-only, SameSite=lax, Secure in prod
- CSP header on every response

## License

MIT (same as parent repo). Research use only.
