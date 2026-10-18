# Common commands. Run `make help`.
SHELL := /bin/bash
ROS_SETUP := source /opt/ros/jazzy/setup.bash

help:
	@echo "make setup        - python venv + core package (laptop)"
	@echo "make test         - run the core unit tests"
	@echo "make simulate     - try the check logic with fake trials (no ROS, no GPU)"
	@echo "make db           - start Postgres + dashboard (docker compose, needs .env)"
	@echo "make demo-data    - fill the DB with fake trials for dashboard work"
	@echo "make results      - print every results table from the trial log"
	@echo "make ros          - build the ROS 2 workspace (needs ROS 2 Jazzy)"
	@echo "make web          - build the dashboard front end"
	@echo "make targets      - write printable checkerboard and ArUco markers"

setup:
	python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

test:
	cd twincheck_core && python -m pytest -q

simulate:
	python scripts/simulate_checks.py --n 200

db:
	docker compose up -d

demo-data:
	python scripts/seed_demo_data.py

results:
	python scripts/results.py --sweep

ros:
	$(ROS_SETUP) && pip install -e twincheck_core && cd ros2_ws && colcon build --symlink-install

web:
	cd dashboard/web && npm install && npm run build

targets:
	python calib/make_targets.py --out calib/targets

.PHONY: help setup test simulate db demo-data results ros web targets
