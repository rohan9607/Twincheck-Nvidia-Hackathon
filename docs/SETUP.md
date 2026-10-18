# Setup

Order: laptop (everyone) → Spark (captain first) → database and dashboard → Isaac Sim → Brev (backup only).

## 1. Laptop (everyone, 30 minutes)

```bash
git clone <team repo> twincheck && cd twincheck
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test && make simulate
```

ROS 2 on the laptop is only needed by whoever writes ROS nodes:
- Ubuntu 24.04: install ROS 2 Jazzy from https://docs.ros.org/en/jazzy/Installation.html, then
  `sudo apt install ros-jazzy-cv-bridge ros-jazzy-moveit python3-colcon-common-extensions` and `make ros`.
- macOS / Windows: use the container: `docker build -f docker/ros.Dockerfile -t twincheck-ros .`
  then `docker run --rm -it --net=host -v $PWD:/twincheck twincheck-ros`.

Check ROS works across two machines: `ros2 run demo_nodes_py talker` on one, `ros2 run demo_nodes_py listener`
on the other, same Wi-Fi, same `ROS_DOMAIN_ID` (pick one number for the team and put it in `.env`).

## 2. CSI VPN (every time you use the Spark)

1. The CSI account password is temporary: change it first (at a CSI computer, or over VPN + Remote Desktop
   to an IP from Shamar Webster, then Ctrl+Alt+End → Change a password).
2. FortiClient VPN ("VPN Only"), SSL-VPN, gateway `vpn.csi.uwm.edu`, port 443.
3. Stuck connecting? Accept the certificate pop-up; if none, email webste63@uwm.edu.

## 3. YAAD Spark workspace

Captain, once (no VPN needed for the website):
1. Sign in to YAAD GPU Lab with your @uwm.edu email, using the 6-digit code (not the link).
2. `ssh-keygen -t ed25519` and paste only the `.pub` line.
3. Start a team named **Wicked**; send the 8-character team code to teammates.

Everyone: join with the code and your own `.pub` key. When it shows Ready, save your login line
(`ssh yt-...@172.30.102.xx`). Full guide: https://www.yaadops.com/guide

On the Spark (on VPN):
```bash
cd /workspace && git clone <team repo> twincheck && cd twincheck
cp .env.example .env && nano .env          # real POSTGRES_PASSWORD and TWINCHECK_DASH_TOKEN (openssl rand -hex 16)
docker compose up -d                       # Postgres (localhost only) + token-protected dashboard on :8090
```

Spark facts that matter:
- It is **arm64** (NVIDIA Grace). Use the NVIDIA PyTorch / Isaac Sim containers from the YAAD guide for GPU work,
  not pip wheels built for x86.
- The team shares one Spark's GPU and memory (100 GB). One **Spark driver** runs heavy jobs; others prepare data and code.
- Long jobs go in `tmux` so a dropped VPN does not kill them.
- **Everything is deleted after the event.** Push code daily; copy datasets, models and `pg_dump` output to the team drive.

## 4. Isaac Sim and the twin

1. Download the Dataset 3 twin from the Innovation Lab data folder into `/workspace/fanuc_er4ia_sim/`.
2. Start Isaac Sim the way the YAAD guide describes (container). Inside it:
   ```bash
   /isaac-sim/python.sh sim/explore_twin.py --usd /workspace/fanuc_er4ia_sim/<main>.usd
   ```
3. Fill `sim/sim_config.yaml` (prim paths, finger joints) and the sizes in `config/twincheck.yaml` from the report.
4. `/isaac-sim/python.sh sim/finger_fit_test.py` → which objects fit the jaws (organizer feedback point 1).

## 5. Brev (backup only)

The Spark does rendering and training; use Brev only if the Spark is busy.
```bash
python scripts/compute_session.py start --machine brev:<name> --gpu "L40S" --purpose "..." --owner <you>
# ... work ...  then STOP THE INSTANCE in the Brev console
python scripts/compute_session.py stop --id <n> --credits <used>
```

## 6. Coding agent

Point it at this repo and the twin folder. Always run what it writes and look at the output (an image, a video,
a test). For YAAD setup an agent can follow https://www.yaadops.com/agents#setup and stop when it needs you.
