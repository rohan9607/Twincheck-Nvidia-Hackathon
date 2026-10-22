#!/bin/bash
# Whole ROS pipeline on a laptop, no Isaac Sim, no arm: real TwinCheck nodes + fake_world.
# Needs ROS 2 Jazzy, the workspace built in ~/tc_ws (docs/SETUP.md), and Postgres up (make db).
#
#   bash scripts/fake_world_run.sh watch fake_watch               # 30 trials, Watch mode
#   bash scripts/fake_world_run.sh guide fake_guide -p n:=12 \
#       -p scenarios:=none,nudged_part,empty_grasp,nudged_empty -p weights:=2,1,1,1
#                                       # Guide, DISARMED: recoverable failures only (a STOP waits for a resume)
#   bash scripts/fake_world_run.sh watch fake_watch -p seed:=7 -p keep_in_bins:=false
#
# Extra args go to fake_world (see the top of twincheck_checker/fake_world.py).
# Rows of the session are deleted first, then results.py prints the tables for it.
MODE=${1:?watch or guide}; TAG=${2:?session tag}; shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
source /opt/ros/jazzy/setup.bash      # before set -u: ROS's setup scripts read unset variables
source ~/tc_ws/install/setup.bash
set -u
set -a; . <(sed 's/\r$//; s/ *#.*$//' "$REPO/.env"); set +a
export TWINCHECK_CONFIG=$REPO/config/twincheck.yaml
export ROS_DOMAIN_ID=${FAKE_DOMAIN_ID:-77}   # keep laptop runs off the team's domain
SHEET=$REPO/ros2_ws/src/twincheck_bringup/config/run_sheet_fake.yaml
OUT=$REPO/runs/fake_world; mkdir -p "$OUT"
cd /tmp   # the repo root has a twincheck_core/ folder that would shadow the installed package

# Nodes left over from an earlier run on this domain would answer this run's events too (duplicate checks,
# stale run sheets). Refuse to start rather than mix runs.
if pgrep -f '[t]c_ws/install/twincheck' > /dev/null; then
    echo "TwinCheck nodes from an earlier run are still alive:"; pgrep -af '[t]c_ws/install/twincheck' | cut -c1-120
    echo "stop them first:  pkill -INT -f '[t]c_ws/install/twincheck'"; exit 1
fi
python3 -c "import os,psycopg,sys; psycopg.connect(os.environ['TWINCHECK_DSN'],autocommit=True).execute('DELETE FROM trials WHERE session_tag=%s',(sys.argv[1],))" "$TAG"
EVENTS=manual; [ "$MODE" = guide ] && EVENTS=guide
# Own process group, so the teardown reaches every node. (A non-interactive shell starts background jobs with
# SIGINT ignored, so signalling only `ros2 launch` left its nodes running into the next run.)
setsid ros2 launch twincheck_bringup twincheck.launch.py env:=twin mode:="$MODE" events:=$EVENTS perception:=false \
    sim_time:=false session_tag:="$TAG" run_sheet:="$SHEET" image_dir:="$OUT/images" > "$OUT/launch_$TAG.log" 2>&1 &
LPID=$!
stop_launch() {
    kill -INT -- -$LPID 2>/dev/null; sleep 3
    kill -TERM -- -$LPID 2>/dev/null; sleep 2
    kill -KILL -- -$LPID 2>/dev/null; wait $LPID 2>/dev/null
}
trap stop_launch EXIT
sleep 4
timeout 600 ros2 run twincheck_checker fake_world --ros-args -p driver:="$MODE" -p run_sheet:="$SHEET" \
    -p session_tag:="$TAG" "$@" > "$OUT/fake_$TAG.log" 2>&1
stop_launch; trap - EXIT
pgrep -f '[t]c_ws/install/twincheck' > /dev/null && echo "WARNING: some TwinCheck nodes survived the teardown"

echo "logs: $OUT/launch_$TAG.log  $OUT/fake_$TAG.log"
grep -nE "Traceback|process has died" "$OUT/launch_$TAG.log" "$OUT/fake_$TAG.log" | head -5
echo "failed checks:"
grep -E "FAIL -" "$OUT/launch_$TAG.log" | sed 's/.*\]: //' | head -40
cd "$REPO" && python3 scripts/results.py --session "$TAG" | sed -n '1,/^Training runs/p'
