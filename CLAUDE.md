# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DPDS (Data Poisoning Detection System) - A Django-based web application for detecting data poisoning and backdoor attacks in CSV datasets. Users upload datasets, run detection algorithms, view poisoned records, sanitize data, and generate reports. The UI is entirely in Simplified Chinese with a dark tech theme.

## Key Commands

```bash
# One-shot initialization (migrates DB, creates test users, registers algorithms, collects static)
python init_system.py

# Development server
python manage.py runserver 0.0.0.0:8000

# Seed detection algorithms into DB (required after fresh migrate)
python manage.py register_algorithms

# Optional: async task worker (falls back to daemon threads if Redis/Celery unavailable)
celery -A DataPoisoningDetection worker -l info
```

There are no tests and no linting configuration in this project.

## Test Accounts

- **Admin**: admin / admin123 (full access including admin panel)
- **User**: user / user123 (regular analyst role)

## Architecture

**Modular Django apps + Vue 3 SPA frontend (dark tech theme)**

```
Browser (Vue3 + Element Plus → static/js/app.js)
    ↓ REST API (JWT auth)
Django REST Framework
    ↓
apps/accounts/     → JWT auth, user management, admin APIs
apps/dpds_datasets/ → Dataset CRUD, upload, preview
apps/preprocessing/ → Data preprocessing (dedup, fill missing, normalize)
apps/detection/     → Detection task management + result storage
apps/defense/       → Data sanitization (remove/relabel/ignore)
apps/reports/       → HTML report generation
apps/audit/         → Audit log middleware + views
    ↓
algorithm_engine/   → Pluggable detection framework (BaseDetector + registry)
    ↓
SQLite (db.sqlite3)  |  Media files (media/)
```

### Core Files

| File | Role |
|------|------|
| `DataPoisoningDetection/settings.py` | Django settings, SQLite/MySQL config |
| `DataPoisoningDetection/urls.py` | Main URL routing |
| `apps/accounts/views.py` | Login, profile, logout, user list |
| `apps/accounts/admin_views.py` | Admin: user management, dashboard, audit logs |
| `apps/dpds_datasets/models.py` | Dataset model (UUID PK, FileField, column_meta) |
| `apps/dpds_datasets/views.py` | DatasetViewSet, upload, preview |
| `apps/detection/models.py` | AlgorithmConfig, DetectionTask, DetectionResult |
| `apps/detection/views.py` | Task CRUD, progress, results, analysis |
| `apps/detection/services.py` | run_detection() orchestrator |
| `apps/defense/services.py` | apply_defense() - remove/relabel/ignore rows |
| `apps/reports/services.py` | generate_report() - HTML report builder |
| `apps/preprocessing/services.py` | run_preprocess() - CSV cleaning |
| `algorithm_engine/base.py` | BaseDetector ABC |
| `algorithm_engine/registry.py` | @register_detector, get_detector(), list_detectors() |
| `algorithm_engine/engine.py` | DetectionEngine.run() - multi-detector orchestrator |
| `static/js/app.js` | Full Vue 3 SPA (14 pages) |
| `templates/index.html` | Single HTML template (Vue mount + all CSS) |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login/` | POST | JWT login |
| `/api/auth/profile/` | GET | Current user info |
| `/api/auth/admin/dashboard/` | GET | Admin dashboard stats |
| `/api/auth/admin/users/` | GET | Admin user list |
| `/api/auth/admin/users/<id>/toggle/` | POST | Toggle user active status |
| `/api/auth/admin/audit-logs/` | GET | Paginated audit logs |
| `/api/datasets/` | GET/POST | List datasets / CRUD |
| `/api/datasets/upload/` | POST | Upload dataset file |
| `/api/datasets/<id>/preview/` | GET | Dataset preview |
| `/api/datasets/<id>/preprocess/` | POST | Run preprocessing |
| `/api/preprocess/<id>/` | GET | Preprocess status/result |
| `/api/detection/tasks/` | GET/POST | List/create detection tasks |
| `/api/detection/tasks/<id>/` | GET | Task detail |
| `/api/detection/tasks/<id>/progress/` | GET | Task progress (Redis) |
| `/api/detection/tasks/<id>/results/` | GET | Detection results |
| `/api/detection/tasks/<id>/analysis/` | GET | Structured analysis |
| `/api/detection/detectors/` | GET | Available detectors metadata |
| `/api/defense/tasks/<id>/apply/` | POST | Apply defense strategy |
| `/api/defense/<id>/download/` | GET | Download clean dataset |
| `/api/reports/` | POST | Generate report |
| `/api/reports/<id>/download/` | GET | Download report |

### Response Format

All APIs return: `{code: 0, msg: "OK", data: {...}}` on success, `{code: 4001, msg: "...", data: null}` on error.

### Detection Methods

| Code | Type | Algorithm |
|------|------|-----------|
| `cleanlab` | label_poison | Confident Learning (cleanlab or self-confidence fallback) |
| `isolation_forest` | anomaly | Isolation Forest |
| `lof` | anomaly | Local Outlier Factor |
| `ks_drift` | distribution | KS test (needs baseline) |
| `mmd_drift` | distribution | MMD test (needs baseline) |
| `spectral_signature` | backdoor | SVD spectral analysis |
| `activation_clustering` | backdoor | (disabled, placeholder) |
| `influence` | influence | (disabled, placeholder) |

## Important Conventions

- **UUID PKs**: All models use `UUIDField(primary_key=True)`.
- **Daemon threads**: Detection tasks run in daemon threads (no Celery required).
- **MD5 deduplication**: Upload checks MD5; duplicates return existing dataset.
- **Column auto-detection**: `label/class/target/y` → label field; `text/content/sentence/input` → text field.
- **Sample size capping**: Detectors cap at 3,000–5,000 rows.
- **Database**: SQLite by default (`USE_SQLITE=True` in `.env`).
- **CORS**: Fully open (`CORS_ALLOW_ALL_ORIGINS = True`).
- **Audit middleware**: Logs all POST/PUT/PATCH/DELETE to `dpds_audit_log`.
