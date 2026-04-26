# Data Retention Policy — amlcombo.org

This document is the binding policy for what data the platform stores,
where, for how long, and who can access it. It exists to satisfy reviewer
concern #4b (post-v0.3) and to give institutional reviewers a single
auditable reference.

**Last reviewed**: 2026-04-26
**Effective**: from kit version v0.4 onward

---

## 1. Data categories

### 1.1 Account data (long-lived)

| Field | Storage | Retention | Encrypted at rest |
|-------|---------|-----------|-------------------|
| User email (login) | PostgreSQL `users.email` | Account lifetime + 2 years after deletion | No (lookup) |
| Password hash (Argon2id) | `users.password_hash` | Account lifetime | One-way hash |
| Display name, institution | `users.{name, institution}` | Account lifetime | No |
| Account creation timestamp | `users.created_at` | Forever (audit) | No |

### 1.2 BYOK LLM keys (researcher-provided)

| Field | Storage | Retention | Protection |
|-------|---------|-----------|------------|
| Encrypted API key | `llm_keys.encrypted_key` | Until user revokes; immediate purge on revoke | Fernet (AES-128-CBC + HMAC-SHA256), key in env var `LLM_KEY_FERNET_SECRET` |
| Last-used timestamp | `llm_keys.last_used_at` | 90 days | No |

The plaintext API key exists in process memory ONLY for the duration of a
single LLM call. It is never written to disk, never logged, never returned
in any API response after the initial /llm-keys POST.

### 1.3 Patient submission data (research-only)

| Field | Storage | Retention | Protection |
|-------|---------|-----------|------------|
| Patient label (de-identified, e.g. `A-2026-001`) | `submissions.patient_label` | User-controlled (delete from `/dashboard`) | None — assumed de-identified per Terms |
| Mutation panel (gene + VAF + codon) | `submissions.input_payload` (JSONB) | User-controlled | At-rest disk encryption (Hetzner LUKS) |
| Karyotype text | `submissions.input_payload` | User-controlled | Same |
| Lab values, demographics | `submissions.input_payload` | User-controlled | Same |
| RNA-Seq counts | `submissions.input_payload` (TEXT, ≤ 5MB) | User-controlled | Same |
| Generated report (MD/HTML/PDF) | `runs/<submission_id>/clinical_report.*` | Same as submission | Filesystem permission `0640` |

**User responsibility (per Terms)**: only de-identified patient labels (e.g.,
`A-2026-001`, `INST-2026-N`). The PHI detector (see §1.4) is a SECONDARY
defense — primary defense is user discipline.

### 1.4 LLM audit log — content-free

| Field | Storage | Retention | Notes |
|-------|---------|-----------|-------|
| `usage_events` rows (`event_type='llm_audit'`) | PostgreSQL | **2 years**, then aggregate-and-purge | Append-only, no PII |

Each `llm_audit` row records:

```jsonc
{
  "feature": "parse_clinical_text_zip",
  "provider": "anthropic",
  "model": "claude-haiku-4-5",
  "input_chars": 4521,
  "tokens_in": 1200,
  "tokens_out": 480,
  "phi_categories_detected": ["NAME_CN", "MRN"],
  "phi_high_severity_count": 2,
  "phi_policy": "redact",
  "success": true
}
```

**It does NOT record**:
- the raw prompt text
- the LLM response body
- the user's API key (already encrypted in `llm_keys`)
- the actual PHI values that were detected (only categories like `"NAME_CN"`)

This means an audit-log breach would expose: usage patterns, cost
attribution, PHI-detection efficacy. It would NOT expose patient data.

### 1.5 Server-side logs (FastAPI / Caddy / Celery)

| Source | Storage | Retention |
|--------|---------|-----------|
| FastAPI access log | `/var/log/web.log` | 14 days, then logrotate purge |
| Caddy reverse-proxy | `/var/log/caddy/` | 14 days |
| Celery task log | `/var/log/celery.log` | 14 days |

These logs record HTTP method + path + status + latency + IP. They do NOT
record request bodies (we explicitly disable body logging in FastAPI).

---

## 2. Third-party processors

| Processor | Data sent | Sent by | Purpose | Their retention |
|-----------|-----------|---------|---------|-----------------|
| **OpenAI** (api.openai.com) | Smart-paste / report-summary prompts (PHI-redacted by default) | User's BYOK call | LLM inference | OpenAI default 30 days unless ZDR; the user's account governs |
| **Anthropic** (api.anthropic.com) | Same as OpenAI | User's BYOK call | LLM inference | Anthropic 30-day input retention; the user's account governs |
| **Cloudflare** (CDN + DNS) | TLS-terminated request metadata | User browser | DDoS / cert / DNS | Cloudflare retention applies |
| **Hetzner** (VPS host) | All data at rest | Server admin | Hosting | Hetzner doesn't access guest content |

**Critical**: when the user enables `phi_policy="passthrough"` (with
`acknowledge_phi=True`), unredacted text reaches the LLM provider. The
provider's retention then applies. We strongly recommend leaving `phi_policy`
at the default `"redact"`.

---

## 3. Data subject rights (GDPR Art. 15-22 / 个保法 第十五条-第十九条)

| Right | How to exercise | Response time |
|-------|-----------------|---------------|
| Access | `GET /api/me/export` returns user data + submissions + audit log | 7 days |
| Rectification | `PATCH /api/me` for account fields; resubmit for patient data | Immediate |
| Erasure | `DELETE /api/me/submissions/<id>` (single) or `DELETE /api/me` (whole account) | Immediate (cascade) |
| Portability | `GET /api/me/export?format=json` | 7 days |
| Restriction | Email `eric@amlcombo.org` to disable processing | 7 days |
| Object | Same; account hard-deleted within 30 days | 30 days |

After account deletion: submissions + reports purged immediately;
`usage_events` rows retained as anonymized aggregates (user_id NULLed) for
2 years for security audit; LLM keys purged immediately.

---

## 4. Cross-border transfer

**Server location**: Hetzner Cloud — Falkenstein, Germany (EU jurisdiction).
**Database**: PostgreSQL on the same Hetzner instance, EU.
**Backups**: Hetzner snapshot, EU only.

**LLM API calls**: cross-border to OpenAI (US) or Anthropic (US) via the
user's BYOK call. The user is the data exporter; we are the processor.

For users in mainland China:
- Per 《个人信息保护法》第三十八条, individual API users sending personal
  information to overseas LLM providers may need CAC approval depending on
  data category and volume.
- Our PHI-redaction default makes outbound prompts PHI-clean by design,
  reducing (but not eliminating) the regulatory exposure.
- Institutional users should consult their compliance office before
  enabling `phi_policy="passthrough"`.

For users in the EU:
- Hetzner DE = EU; local processing is GDPR Art. 44 compliant.
- Outbound LLM (US) is covered by OpenAI / Anthropic's EU standard
  contractual clauses (SCCs). User is data controller for the BYOK call.

---

## 5. Breach notification

If we detect unauthorized access to any data category in §1, we will:

1. Investigate root cause within 24 hours
2. Notify affected users within 72 hours via the email on file
3. File a public incident report at https://amlcombo.org/security
4. Notify CAC (if Chinese users affected) and EU DPA (if EU users affected)
   per applicable regulation

---

## 6. Audit access

The PostgreSQL `usage_events` table is the SINGLE long-term audit record.
It is queryable by:

- The user themselves: `GET /api/me/audit-log` returns their own events
- The platform operator (Eric Tom): for security investigation only;
  every operator query is logged in a separate `operator_access_log` table

External auditors (institutional compliance officers) can request access
via `eric@amlcombo.org` with a signed NDA and a defined scope. The audit
log contains no patient content (see §1.4) so this is low-risk.

---

## 7. Changelog

| Date | Change |
|------|--------|
| 2026-04-26 | Initial policy (v0.4) — created in response to clinical-reviewer concern #4b |
