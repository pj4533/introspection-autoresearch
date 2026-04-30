#!/usr/bin/env bash
# Re-compute Phase 4 site JSON files and commit them so Vercel
# picks up the latest data. Safe to run while the dream loop is
# still going — it just reads the DB.

set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate

.venv/bin/python scripts/compute_forbidden_map.py
.venv/bin/python scripts/export_dream_walks.py
.venv/bin/python scripts/compute_attractors.py

echo
echo "Phase 4 site data refreshed:"
ls -la web/public/data/forbidden_map.json web/public/data/dream_walks.json web/public/data/attractors.json

echo
echo "Commit + push to deploy?"
echo "  git add web/public/data/ && git commit -m 'phase4: refresh site data' && git push"
