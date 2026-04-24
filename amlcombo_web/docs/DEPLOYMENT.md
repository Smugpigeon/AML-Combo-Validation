# Deployment guide — amlcombo.org

End-to-end walkthrough from a fresh laptop to a live public site. Skip to step-numbers you've already done.

**Estimated time**: 90 minutes. **Recurring cost**: ~¥80/month (Hetzner VPS €4.51 + domain amortized).

---

## Architecture recap

```
Browser → Cloudflare (free SSL/DDoS) → VPS (Hetzner) → Caddy (reverse proxy)
                                                     ↓
                                            ┌──────────────────┐
                                            │  web (FastAPI)   │
                                            │  worker (Celery) │
                                            │  db (Postgres)   │
                                            │  redis           │
                                            └──────────────────┘
```

All 5 containers run on a single Hetzner CX22 (€4.51/mo, 4 GB RAM / 2 vCPU / 40 GB SSD).

---

## Step 1 — Rent a VPS (Hetzner Cloud)

### Why Hetzner?
- Cheapest reliable EU/US VPS (€4.51/mo = ~¥35/mo for CX22)
- Hourly billing, delete anytime
- Good uptime, fast network
- Alternative: DigitalOcean Basic $24/mo (3x the price but easier UI; your call)

### Rental steps

1. Go to <https://www.hetzner.com/cloud> → **Sign up** (takes ~10 min; they verify with a credit-card charge of €1 refunded)
2. Once logged into [Hetzner Cloud Console](https://console.hetzner.cloud):
   - Click **New project** → name it `amlcombo`
   - Click **Add server**
   - **Location**: pick one close to your users (Ashburn VA for US, Helsinki/Falkenstein for EU, Hillsboro OR for US-West)
   - **Image**: Ubuntu 24.04
   - **Type**: **CX22** (shared CPU, 4 GB RAM, 2 vCPU, 40 GB SSD) — €4.51/mo
   - **SSH key**: paste your public key (if you don't have one: `ssh-keygen -t ed25519` on your laptop, then `cat ~/.ssh/id_ed25519.pub`)
   - **Name**: `amlcombo-prod`
   - Click **Create & Buy now** → ~30 seconds
3. Copy the server's **IPv4 address** from the console (you'll need it in Step 3)
4. Test SSH: `ssh root@<vps-ip>` — should log in without password.

### Optional: enable automated backups

In Hetzner console → server → **Backups** → **Enable** → +20% to the monthly cost (~€0.90/mo). Gives you 7 daily snapshots. Worth it.

---

## Step 2 — Point `amlcombo.org` at the VPS

You said the domain is at **Namecheap**. We'll route through Cloudflare for free SSL + DDoS.

### 2a. Create a free Cloudflare account
1. <https://dash.cloudflare.com/sign-up> — verify email
2. **Add site** → enter `amlcombo.org` → **Free** plan
3. Cloudflare will show you **2 nameservers** like `adi.ns.cloudflare.com` and `kurt.ns.cloudflare.com`. Keep this tab open.

### 2b. Change nameservers at Namecheap
1. Log in to Namecheap → **Domain List** → `amlcombo.org` → **Manage**
2. Under **Nameservers**, change from "Namecheap BasicDNS" to **Custom DNS**
3. Paste the 2 Cloudflare nameservers from 2a
4. Click the green ✓ to save
5. Back in Cloudflare, click **Done, check nameservers**. Usually ready in 10 min, can take up to 24 h.

### 2c. Add DNS records (in Cloudflare, not Namecheap)
In Cloudflare → **DNS** → **Records** → **Add record**:

| Type | Name | Content | Proxy status |
|------|------|---------|--------------|
| A    | `@`  | `<your-vps-ipv4>` | **Proxied** (orange cloud) |
| A    | `www` | `<your-vps-ipv4>` | **Proxied** |

### 2d. SSL/TLS settings
In Cloudflare → **SSL/TLS** → **Overview**:
- Set to **Full (strict)** — Caddy on your VPS will get a real Let's Encrypt cert, and Cloudflare will trust it.

### 2e. Test DNS propagation
`dig amlcombo.org @1.1.1.1` should return your VPS IP (proxied through Cloudflare so you may see a Cloudflare IP, that's fine).

---

## Step 3 — Prepare the VPS

SSH in as root:
```bash
ssh root@<vps-ip>
```

### 3a. Create a non-root user and lock down SSH
```bash
adduser deploy
usermod -aG sudo deploy
mkdir /home/deploy/.ssh
cp ~/.ssh/authorized_keys /home/deploy/.ssh/
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys

# Disable password auth + root login
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Test: from another terminal
# ssh deploy@<vps-ip>       should work
# ssh root@<vps-ip>         should refuse
```

### 3b. Install Docker + Docker Compose
```bash
sudo apt update && sudo apt upgrade -y
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker deploy
newgrp docker              # refresh group in current shell
docker --version           # verify
docker compose version     # verify
```

### 3c. Firewall (ufw)
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp       # SSH
sudo ufw allow 80/tcp       # HTTP (for LE cert renewal)
sudo ufw allow 443/tcp      # HTTPS
sudo ufw allow 443/udp      # HTTP/3
sudo ufw enable
```

---

## Step 4 — Clone the repo and configure

As the `deploy` user:
```bash
cd ~
git clone https://github.com/ericktom/AML-combo-validation.git
cd AML-combo-validation/amlcombo_web
cp .env.example .env
```

### 4a. Generate secrets (still on the VPS)
```bash
# JWT_SECRET
python3 -c "import secrets; print(secrets.token_urlsafe(48))"

# FERNET_KEY (DO NOT REGENERATE LATER — it decrypts user-stored LLM keys)
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# If python3 doesn't have cryptography: pip install cryptography --user

# POSTGRES_PASSWORD
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 4b. Edit `.env`

```bash
nano .env
```

Paste in the 3 secrets you just generated. Also set `DOMAIN=amlcombo.org`.

**IMPORTANT**: back up `FERNET_KEY` somewhere safe (password manager, encrypted disk). If you lose it, every user's stored LLM key becomes un-decryptable, and you must have them re-add their keys.

### 4c. Start the stack
```bash
cd ~/AML-combo-validation/amlcombo_web
docker compose up -d
```

First build takes ~10 minutes (installing PyTorch + Chromium). Watch with:
```bash
docker compose logs -f
```

When you see `Application startup complete` and `celery@... ready.`, you're live.

---

## Step 5 — First login

1. Visit `https://amlcombo.org` in your browser
2. Cloudflare shows up-to-date SSL cert? Good
3. Click **Sign up** → create your account
4. You're in! Go to `/api-keys` → generate your first key → save it

### Optional: make yourself an admin via shell
If you want to verify a user's email manually (there's no email verification in MVP):
```bash
docker compose exec web python scripts/create_admin.py
```

---

## Step 6 — Smoke test end-to-end

On your laptop:
```bash
# Get your API key from /api-keys page
export API_KEY=ac_live_xxxxxxxxxxxxxxxxxxxxxxxx

# Sample patient JSON
cat > /tmp/patient.json <<'EOF'
{
  "patient_label": "TEST-001",
  "age": 45, "sex": "female",
  "is_initial_diagnosis": true,
  "karyotype_text": "46,XX[20]",
  "mutations": [{"gene":"FLT3","is_ITD":true,"allelic_ratio":0.62,"vaf":0.45}],
  "wbc": 95.0, "platelet": 32.0, "hemoglobin": 8.5, "ldh": 1240.0
}
EOF

# Sample RNA counts (replace with real data in prod)
echo "symbol,count" > /tmp/rna.csv
# ... append ~5000 gene symbols + counts

# Submit
curl -X POST https://amlcombo.org/api/v1/patients \
  -H "Authorization: Bearer $API_KEY" \
  -F "payload_json=<@/tmp/patient.json" \
  -F "rna_counts=@/tmp/rna.csv"

# Poll status
curl https://amlcombo.org/api/v1/patients \
  -H "Authorization: Bearer $API_KEY"

# Download report (once status=done)
curl https://amlcombo.org/api/v1/reports/<submission_id>/pdf \
  -H "Authorization: Bearer $API_KEY" \
  -o report.pdf
```

---

## Step 7 — Ongoing operations

### Check logs
```bash
docker compose logs -f web       # FastAPI
docker compose logs -f worker    # Celery
docker compose logs -f caddy     # reverse proxy
```

### Restart services
```bash
docker compose restart web worker
```

### Deploy new version
```bash
cd ~/AML-combo-validation
git pull
cd amlcombo_web
docker compose build
docker compose up -d
```

### Manual DB access
```bash
docker compose exec db psql -U amlcombo
# \dt    list tables
# select count(*) from users;
```

### Backup the database
```bash
docker compose exec db pg_dump -U amlcombo amlcombo > backup_$(date +%F).sql
# Off-site: scp to your laptop or a different cloud
```

### Disk usage check
```bash
docker system df                                         # all containers
du -sh /var/lib/docker/volumes/amlcombo_storage_data/   # user uploads
```

---

## Cost summary (live checklist)

| Item | Provider | Monthly | Annual |
|------|---------|---------|--------|
| VPS (CX22, 4 GB) | Hetzner | €4.51 | €54 |
| VPS backups (+20%) | Hetzner | €0.90 | €11 |
| Domain | Namecheap | — | $18 (already paid) |
| DNS + SSL + DDoS | Cloudflare Free | €0 | €0 |
| **Total recurring** | | **~€5.5** | **~€65** (≈ ¥500/year) |

LLM costs are **$0 to you** — users pay their own OpenAI/Anthropic bill (BYOK model).

---

## Security considerations

- **FERNET_KEY** unlocks all stored user LLM keys. Back it up off-server.
- **.env** is gitignored. Never commit it.
- **Postgres port 5432** is NOT exposed externally (only on the internal Docker network). Good.
- **Cloudflare origin cert** (optional upgrade): If you want end-to-end encryption that also validates Cloudflare → your server, generate an Origin Certificate in Cloudflare and use it in the Caddyfile. MVP uses Let's Encrypt which is simpler.
- **Rate limiting**: Built in — 10 patients/day per user. Adjust `DAILY_PATIENT_SUBMIT_LIMIT` in `.env`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Caddy can't get SSL cert | Cloudflare proxy must use **Full (strict)**, not Flexible. Also ensure port 80 is open. |
| `502 Bad Gateway` | Web container crashed. `docker compose logs web` for traceback. |
| Celery tasks stuck in `queued` | Worker dead. `docker compose restart worker`. |
| Upload fails with 413 | `MAX_UPLOAD_BYTES` in `.env` + `max_size` in `Caddyfile` must agree. |
| PDF rendering fails | Chromium missing fonts. Check `docker exec -it web chromium --version` works. |

---

## Next steps after MVP

1. Add **email verification** (Resend.com free tier 3k/mo)
2. Add **Stripe subscriptions** if you want paid tiers
3. Add **Sentry** for error tracking (`SENTRY_DSN` env var — integration is 2 lines)
4. Add **S3-compatible object storage** (Cloudflare R2 — 10 GB free) for RNA-Seq files if storage_data volume gets big
5. Set up **uptime monitoring** (UptimeRobot free, pings every 5 min)
