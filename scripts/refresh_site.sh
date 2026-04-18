#!/bin/bash
# Refresh the public Next.js site with the latest data and redeploy to Vercel.
#
# Loop mode: runs every REFRESH_INTERVAL seconds (default 900 = 15 min).
# Single-shot mode: REFRESH_ONCE=1 ./scripts/refresh_site.sh
#
# The loop:
#   1. Runs scripts/export_for_web.py (reads SQLite, writes web/public/data/*.json)
#   2. If any JSON files changed, runs `vercel --prod --yes` to redeploy
#
# Launch:
#   nohup ./scripts/refresh_site.sh > logs/refresh.log 2>&1 &
# Stop:
#   pkill -f "refresh_site.sh"

set -euo pipefail
cd "$(dirname "$0")/.."

REFRESH_INTERVAL="${REFRESH_INTERVAL:-900}"
REFRESH_ONCE="${REFRESH_ONCE:-}"

mkdir -p logs

do_refresh() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] exporting..."

    # Snapshot the content-bearing files (excluding summary.json which always
    # ticks last_updated even without new data).
    local before_hash after_hash
    before_hash=$(
        cat web/public/data/phase2_leaderboard.json \
            web/public/data/phase2_activity.json \
            web/public/data/detections.json \
            web/public/data/layer_curve.json \
            web/public/data/abliteration_comparison.json 2>/dev/null \
        | shasum | awk '{print $1}'
    )

    .venv/bin/python scripts/export_for_web.py --quiet

    after_hash=$(
        cat web/public/data/phase2_leaderboard.json \
            web/public/data/phase2_activity.json \
            web/public/data/detections.json \
            web/public/data/layer_curve.json \
            web/public/data/abliteration_comparison.json 2>/dev/null \
        | shasum | awk '{print $1}'
    )

    if [[ "$before_hash" != "$after_hash" ]]; then
        echo "[$ts] data changed — deploying..."
        # Capture the new deployment URL so we can alias it.
        local deploy_out
        deploy_out=$(cd web && vercel --prod --yes --no-color 2>&1)
        echo "$deploy_out" | tail -5
        # Vercel's new prod deploys are NOT auto-aliased to the clean project
        # domain — we have to do it ourselves after each deploy.
        local deploy_url
        deploy_url=$(echo "$deploy_out" | grep -oE 'did-the-ai-notice-[a-z0-9]+-pjs-projects-3e91f18c\.vercel\.app' | head -1)
        if [[ -n "$deploy_url" ]]; then
            echo "[$ts] aliasing $deploy_url -> did-the-ai-notice.vercel.app"
            (cd web && vercel alias set "$deploy_url" did-the-ai-notice.vercel.app --no-color 2>&1 | tail -2)
        else
            echo "[$ts] WARNING: could not parse deployment URL from vercel output"
        fi
        echo "[$ts] deploy complete."
    else
        echo "[$ts] no meaningful change — skipping deploy."
    fi
}

if [[ -n "$REFRESH_ONCE" ]]; then
    do_refresh
    exit 0
fi

echo "=== refresh loop started at $(date) interval=${REFRESH_INTERVAL}s ==="
while true; do
    do_refresh || echo "[$(date '+%H:%M:%S')] refresh failed — will retry"
    sleep "$REFRESH_INTERVAL"
done
