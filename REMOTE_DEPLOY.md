# Remote Deploy — shared host, behind a reverse proxy

> **Goal:** ssh the project to a remote server that **already runs other Docker projects**, bring
> it up **in parallel** (no port/volume clashes), and serve the **dashboard** behind your existing
> reverse proxy. The demo is operator-triggered over SSH; viewers just watch the dashboard.
>
> Companion: [`INFRASTRUCTURE.md`](./INFRASTRUCTURE.md) (how it's wired), [`DEMO_RUNBOOK.md`](./DEMO_RUNBOOK.md) (local demo).

---

## 1. How parallel-safety is achieved

| Clash risk on a shared host | How it's avoided |
|---|---|
| **Host ports** (5432/8080/8000/8501/9000/5000) | All bindings are env-driven. The remote `.env` sets `BIND_HOST=127.0.0.1` + unique offset ports, so nothing publishes on `0.0.0.0` or on a port the other project uses. |
| **Volume / network names** | `COMPOSE_PROJECT_NAME=brewleaf` → volumes/networks become `brewleaf_*`, isolated from other stacks. |
| **`localhost:5432` needed by the demo** | The `*_remote.sh` scripts run every step **in-container** (`docker exec sentiment-airflow-scheduler`, reaching `postgres:5432` on the compose network) — no host venv, no host DB port. |

Net result: on the server, **only `127.0.0.1:<DASHBOARD_HOST_PORT>` matters**, and your reverse proxy
forwards to it. Everything else is reachable only via SSH tunnel.

---

## 2. One-time setup

**a. Copy the project to the server** (exclude local junk; the champion model is sent separately):
```bash
rsync -av --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
      ./  you@server:/srv/brewleaf/
```

**b. Copy the champion model** (gitignored, ~6 MB — never arrives via git):
```bash
scp models/artifacts/champion_baseline_v3.pkl you@server:/srv/brewleaf/models/artifacts/
```

**c. Configure env** on the server:
```bash
cd /srv/brewleaf
cp infra/remote.env.example .env
# edit .env: change the *_strong passwords; set free host ports if 5xxxx collide
```

**d. Build + start** (binds to 127.0.0.1, isolated project name — safe alongside other stacks):
```bash
docker compose up -d --build
```

**e. Seed the normal day** (in-container — no host deps):
```bash
bash scripts/demo_up_remote.sh
```

---

## 3. Point your reverse proxy at the dashboard

The dashboard listens on `127.0.0.1:${DASHBOARD_HOST_PORT}` (default `58501`). Add a vhost. Streamlit
needs **WebSocket** upgrade.

**Caddy:**
```
brewleaf.yourdomain.com {
    reverse_proxy 127.0.0.1:58501
}
```

**nginx:**
```nginx
location / {
    proxy_pass http://127.0.0.1:58501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```
Reload the proxy → dashboard live at your domain (HTTPS handled by the proxy).

---

## 4. ▶ Demo-day runbook (over SSH)

**Pre-flight:**
- [ ] `docker compose ps` → all `sentiment-*` healthy
- [ ] `REQUIRE_CHAMPION=1 bash scripts/demo_up_remote.sh` (hard-stops if the champion `.pkl` is missing)
- [ ] dashboard loads at your domain (normal day, ~20% negative)
- [ ] *(recommended)* stop the 6h observer overwriting the spike:
      `docker exec sentiment-airflow-scheduler airflow dags pause evaluate_and_monitor`

**Live (two commands):**
```bash
bash scripts/demo_up_remote.sh      # reset to a clean normal day (~20% negative)
bash scripts/demo_spike_remote.sh   # inject the spike -> ~51% negative, drift recorded, gate blocked
```
Viewers see the dashboard flip to the red alert within ~5s (auto-refresh).

**Reset:** `bash scripts/demo_up_remote.sh`.

---

## 5. Admin UIs (not public — via SSH tunnel)

```bash
ssh -L 58080:127.0.0.1:58080 -L 55001:127.0.0.1:55001 -L 59001:127.0.0.1:59001 you@server
# laptop: Airflow http://localhost:58080 · MLflow http://localhost:55001 · MinIO http://localhost:59001
```

---

## 6. Secrets & hardening

- Change `POSTGRES_PASSWORD`, `AWS_SECRET_ACCESS_KEY`, `AIRFLOW_ADMIN_PASSWORD` in `.env`.
- Keep `ADMIN_TOKEN` unset unless you use `POST /reload`.
- Only 80/443 (proxy) + 22 (SSH) need to be open in the server firewall; every service binds to loopback.

---

## 7. Persistence & troubleshooting

- Data lives in named volumes `brewleaf_pg_data` / `brewleaf_minio_data` — survive `docker compose down`,
  cleared by `down -v`.
- Stop: `docker compose down` (data persists). The other projects on the host are untouched.

| Symptom | Fix |
|---|---|
| `port is already allocated` | another project uses that host port — change the `*_HOST_PORT` in `.env` |
| Dashboard spins / no live update | reverse proxy isn't forwarding WebSockets — fix upgrade headers |
| Weak/odd numbers | champion `.pkl` missing → `REQUIRE_CHAMPION=1 bash scripts/demo_up_remote.sh` |
| Spike not on the drift tile | a clean observer run overwrote it — pause `evaluate_and_monitor` (§4) and re-run the spike |
