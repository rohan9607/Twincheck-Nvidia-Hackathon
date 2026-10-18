-- TwinCheck trial database. Loaded automatically by docker compose on first start.

CREATE TABLE IF NOT EXISTS trials (
  trial_id            SERIAL PRIMARY KEY,
  started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  env                 TEXT NOT NULL CHECK (env IN ('twin', 'real')),
  mode                TEXT NOT NULL CHECK (mode IN ('baseline', 'watch', 'guide')),
  ladder_rung         TEXT NOT NULL DEFAULT 'fixed_pose',
  object_class        TEXT NOT NULL,
  unseen_object       BOOLEAN NOT NULL DEFAULT false,
  induced_failure     TEXT CHECK (induced_failure IN ('empty_grasp', 'removed_mid_grasp', 'nudged_part')),
  seed                INTEGER,
  model_version       TEXT,
  threshold_mm        REAL,
  session_tag         TEXT,
  ros_trial_id        INTEGER,
  pick_point          TEXT,
  pick_success        BOOLEAN,           -- ground truth: twin runner, or the Recorder on the real arm
  place_error_mm      REAL,
  cycle_time_s        REAL,
  retried             BOOLEAN NOT NULL DEFAULT false,
  resync_max_shift_mm REAL,
  notes               TEXT
);

CREATE TABLE IF NOT EXISTS checks (
  check_id         SERIAL PRIMARY KEY,
  trial_id         INTEGER NOT NULL REFERENCES trials(trial_id) ON DELETE CASCADE,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  check_point      TEXT NOT NULL CHECK (check_point IN ('pre_pick', 'post_grasp', 'post_place')),
  passed           BOOLEAN NOT NULL,
  position_err_mm  REAL,
  yaw_err_deg      REAL,
  reason           TEXT,
  action           TEXT,
  urgent           BOOLEAN DEFAULT false,
  expected_img     TEXT,
  observed_img     TEXT,
  sync_gap_ms      REAL,
  expected_source  TEXT
);
CREATE INDEX IF NOT EXISTS checks_trial ON checks(trial_id);

CREATE TABLE IF NOT EXISTS training_runs (
  run_id           SERIAL PRIMARY KEY,
  started_at       TIMESTAMPTZ DEFAULT now(),
  model            TEXT NOT NULL,
  dataset          TEXT NOT NULL,
  epochs           INTEGER,
  gpu              TEXT,
  runtime_min      REAL,
  credits_used     REAL,
  synth_score      REAL,
  real_recall      REAL,
  real_pos_err_mm  REAL,
  real_yaw_err_deg REAL,
  weights_path     TEXT,
  notes            TEXT
);

CREATE TABLE IF NOT EXISTS compute_sessions (
  session_id   SERIAL PRIMARY KEY,
  started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  stopped_at   TIMESTAMPTZ,
  machine      TEXT NOT NULL,            -- brev:<instance> | spark | laptop
  gpu          TEXT,
  purpose      TEXT,
  credits_used REAL DEFAULT 0,
  owner        TEXT
);

-- One row per trial with what TwinCheck said: the basis of every results table.
CREATE OR REPLACE VIEW trial_summary AS
SELECT t.*,
       bool_or(NOT c.passed)                                          AS flagged,
       bool_or(NOT c.passed AND c.check_point = 'pre_pick')           AS flagged_pre_pick,
       bool_or(NOT c.passed AND c.check_point = 'post_grasp')         AS flagged_post_grasp,
       bool_or(NOT c.passed AND c.check_point = 'post_place')         AS flagged_post_place,
       max(c.position_err_mm) FILTER (WHERE c.check_point = 'pre_pick') AS pre_pick_err_mm,
       count(c.check_id)                                              AS n_checks
FROM trials t LEFT JOIN checks c ON c.trial_id = t.trial_id
GROUP BY t.trial_id;
