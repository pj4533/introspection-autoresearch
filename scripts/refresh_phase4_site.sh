#!/usr/bin/env bash
# Re-compute Phase 4 site JSON, commit, push, deploy + alias to
# canonical URL. One command refreshes the live site.
#
# Safe to run while the dream loop is going — it just reads the DB
# (the loop auto-refreshes JSON in place every cycle anyway, but
# this also commits + deploys).
#
# Pass --no-deploy to only refresh JSON (no commit / no Vercel push).

set -e
cd "$(dirname "$0")/.."

DEPLOY=true
if [[ "$1" == "--no-deploy" ]]; then
  DEPLOY=false
fi

source .venv/bin/activate

echo "=== Refreshing JSON ==="
.venv/bin/python scripts/compute_forbidden_map.py
.venv/bin/python scripts/export_dream_walks.py
.venv/bin/python scripts/compute_attractors.py

if [[ "$DEPLOY" == "false" ]]; then
  echo
  echo "Phase 4 site data refreshed (no deploy):"
  ls -la web/public/data/forbidden_map.json web/public/data/dream_walks.json web/public/data/attractors.json
  exit 0
fi

echo
echo "=== Committing + pushing ==="
if git diff --quiet web/public/data/; then
  echo "No JSON changes to commit."
else
  git add web/public/data/
  git commit -m "phase4: refresh site data $(date -u +%Y-%m-%dT%H:%MZ)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
  git push origin main
fi

echo
echo "=== Vercel deploy + alias ==="
cd web
DEPLOY_URL=$(npx vercel --yes --prod 2>&1 | grep -oE "https://[a-z0-9-]+-pjs-projects-[a-z0-9-]+\.vercel\.app" | head -1)
if [[ -z "$DEPLOY_URL" ]]; then
  echo "Could not extract deploy URL — check vercel CLI output manually."
  exit 1
fi
echo "Deployed: $DEPLOY_URL"
npx vercel alias set "$DEPLOY_URL" did-the-ai-notice.vercel.app

echo
echo "Live: https://did-the-ai-notice.vercel.app/"
