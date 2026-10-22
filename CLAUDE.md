# CLAUDE.md: TwinCheck project context (read this first)

This file is the complete history and context of the TwinCheck project, written so a coding agent
(Claude Code) or a new teammate can pick up development without the original planning chat.
Last updated: Oct 20, 2026 (section 1b: what the organizers' cell platform changed).

---

## 0. TL;DR

- **Event:** Rockwell Automation + NVIDIA Innovation Lab, Fall 2026: a 4-week physical-AI hackathon
  run by the MSOE AI Club with UWM's Connected Systems Institute (CSI).
- **Team:** Wicked. Captain: Rohan (UWM M.S. CS, software engineer at GE HealthCare). Two teammates with an
  IST (information sciences and technology) background, new to robotics.
- **Dataset / challenge:** Dataset 3 (Track 3, "Act on it: teach the robot"), Grasping and Sorting, on a standalone
  FANUC ER-4iA arm with an **SMC MHZL2-16D2** two-finger gripper (jaws open only **12 mm**), a table camera
  (overhead, from the front), a side camera, a wrist camera, and an Isaac Sim digital twin (the organizers' starter v0.5).
- **Real arm = the organizers' platform** (section 1b): our `policy.py` answers `act(obs)` over the network during
  booked 30-minute slots; an operator sets up and scores every rollout. `cell_policy/` is TwinCheck in that form.
- **Project:** **TwinCheck**: "Sim-trained grasping that verifies every pick." Cameras watch every pick, the
  digital twin defines what *should* happen, and mismatches raise an alert (Watch mode) or trigger a
  corrected retry (Guide mode). The detector is trained only on twin-rendered images.
- **Key dates:** final submission **Sun Nov 1, 11:59 PM CT** (about 5-minute video, one-page summary, repo
  link, references). Finals **Thu Nov 5, 5–8 PM, MSOE Diercks Hall**.
- **Today (Oct 17):** the team plan was submitted and reviewed, Spark access granted, and the repo scaffolded and
  verified (core logic, DB, dashboard, calibration). Nothing has run on the twin or the real arm yet.
  About 5 days were lost to an exam; the schedule below is compressed accordingly.
- **Start here in code:** `make test`, `make simulate`, then `docs/WORKFLOW.md` step by step.

---

## 1. The hackathon (from the organizer pages)

**Sponsors and organizers:** Rockwell Automation, NVIDIA, MSOE AI Club, UWM Connected Systems Institute (CSI).
Teams, repos and submissions live in the ALL AI Network event workspace; connecting GitHub there auto-invites
you to the team repo, which Rockwell and NVIDIA can view.

**The testbed:** CSI runs a high-mix, low-volume demo plant (Rockwell hardware) that fills vials with
mixed-color fluid. A separate FANUC robot setup (Dataset 3) moves and sorts blocks and other objects. It is
**not connected** to the vial line.

**Three datasets (pick one; focus is recommended):**
1. Final Vial Inspection: missing or crooked caps, damaged vials, empty positions (overhead camera, 2×30-slot
   fixtures). Suggested baseline: TAO Visual ChangeNet with a golden image; missed defects weigh more than false alarms.
2. Fluid Mixing Station 3: color and fill problems from recipe carryover; three synchronized cameras; only 30
   labeled vials; leave-one-recipe-out evaluation suggested.
3. **Grasping and Sorting (ours):** use multiple cameras plus the twin to make the FANUC arm grasp and sort reliably.
   The brief: fixed-motion arms only work if objects arrive in the same spot and cannot adapt to clutter,
   changed objects or imperfect grasps. Suggested sequence: make the existing object and fingers work in sim →
   measure success over repeated trials and varied starts → add one variation at a time (orientation, object
   type, clutter, lighting) → compare with a few physical trials → only then change policy, perception or
   fingers. Keep a log; "a video alone does not show reliability."

**Core rule for all datasets:** real line data is for looking and testing, **not training**. Training data
must come from the digital twin (that is how a high-mix, low-volume line has to work: new products have no
real images yet).

**Dataset 3 safety rules:** do not send commands to the physical arm or change its controller configuration
without the plant manager and assigned mentor. No printed finger on the arm until reviewed for clearance
and safe operation. Follow the "Collect and Test Your Own Data" guide before physical time.

**What judges expect (scoring):**
1. Solution demonstrated **operating on real hardware** (the CSI line or the FANUC arm; schedule a supervised test).
2. A **specific problem** (defect type, failure mode, slow manual step) and why it matters.
3. **Business case for other manufacturers:** what the problem costs today (scrap, rework, downtime, labor),
   what the solution saves, and what adoption takes (hardware, setup time, changeovers, operator trust).
4. **Measured results against a baseline**, with clarity about where it still fails.

Judges represent Rockwell, NVIDIA, MSOE, and UWM; detailed weights are not final.

**What Rockwell said it values:** faster, more reliable changeovers on lines making many variants with fewer
experienced people; operators who understand and trust the AI; results that hold up on real equipment with
clear limits over clean demos. At kickoff, Rockwell also stressed that customers run **decades-old (about 40-year-old)
equipment** and want **scalability without reprogramming** for new products.

**What NVIDIA wants:** proof that models developed and validated in a physically based twin (Omniverse /
Isaac Sim) deploy to the real floor; synthetic data where real data is scarce.

**Organizers also said** (relayed by Rohan): do not rebuild a solution that already exists. We interpret this
as: no off-the-shelf demo presented as the contribution. We are honest about prior art and claim a specific
combination plus measured results (see section 4).

**Prizes:** $5,000 cash pool, RTX 5080, RTX 5070, sponsor swag; the split is TBD.

**Schedule:** kickoff Oct 8 (done) · team plan due Oct 14 (submitted) · final submission Nov 1 · finals Nov 5.

---

## 1b. The organizers' cell platform (read Oct 20; printouts of hackathon.sidclab.org are in the repo root)

The SIDC Lab (UWM) runs the real arm as a remote evaluation service. **This replaces our earlier assumptions**
(ROS 2 + FANUC driver, controller access, Watch on an existing robot program, our own camera calibration).
- **How it works:** client package `github.com/SIDCLab/PhysicalAI-Hackathon-2026` (uv, Linux/macOS; Windows works
  with `PYTHONIOENCODING=utf-8`). We edit `policy.py`; the server sends `obs` and our `act(obs)` returns up to 100
  target rows; the cell plays them and enforces its limits. `uv run run.py --mock` checks formats on a local mock
  (no kinematics, synthetic images), `--submit` sends `submission/submission.yaml` + `placement.yaml`, then we book
  a **30-minute slot** in the Calendar tab; a run is **10 rollouts** by default.
- **Observation:** `images` from `table`, `side`, `wrist` (any subset, up to 1920×1080, png or jpeg); optional depth
  (table 640×400, wrist 848×480, mm); `joints` J1–J6 deg (twin convention, sim J3 = controller J3 + J2) + gripper
  0/1; `tcp_pose` fingertip `[x,y,z,w,p,r]` in UF0; `trial` with the instruction.
- **Action:** `"joints"` `[J1..J6, gripper]` or `"tcp"` `[x,y,z,w,p,r,gripper]`, each row = one 0.1 s step.
  UF0: x, y = robot base, z_UF0 = z_base − 330 mm, work surface about −335 mm. `w,p,r = −180,0,R` = fingers down.
  The gripper only opens/closes (≥0.5 closes), after the arm reaches that row. TCP box x 150–650, y ±450,
  z −334..300 mm; **beyond about 0.55 m reach the controller faults and the rollout ends**. The real arm follows
  its targets **about 0.8 s behind** (the twin does not). The policy cannot end a rollout: hold still when done.
- **Scoring:** the operator scores each rollout as a whole, success or failure; interventions and faults are noted;
  successful rollouts record time to success. Leaderboard: success rate with **Wilson 95% interval**, ties by fewer
  faults. Every rollout is recorded as a **LeRobotDataset** (3 cameras + robot state at 10 Hz), downloadable with
  `--download` (real data: test and analysis only, never training).
- **Hardware:** SMC MHZL2-16D2 with custom aluminium fingers 76 × 12 × 6 mm with a hooked tip; **6 mm travel per
  jaw: straight faces 0–12 mm apart**, hooked tips about 4–16 mm. TCP (UTOOL1) is 293 mm from the J6 origin.
  Objects (CAD on the Resources page): M6×30 countersunk screws, **10 mm and 16 mm cubes**, bins 50×50×55 mm
  (35×35 mm pocket), vial Ø27.18×203 mm, cap Ø31.75 mm. **Only the 10 mm cubes and the screws fit the jaws.**
  Custom fingers/objects go under `custom_hardware` in submission.yaml and are reviewed. Default scene: 8 screws
  and the yellow bin.
- **Calibration is provided:** `calibration/cameras.json` + `geometry.py` (`pixel_to_base`, `depth_to_base`); the
  starter's sim cameras already use it, so twin pixels and real pixels line up. Our ArUco/homography step is
  not needed for the cell.
- **Starter v0.5** (`fanuc_er4ia_hackathon_starter_v05/`, untracked): builds the Isaac Sim 6.0 cell in code
  (`cell.py`, `scene_config.py`), task = stack six 10 mm cubes, perception = Cosmos + image fit. It shrinks the vial
  to 17 × 140 mm (real: set `VIAL_RADIUS = 0.02718/2`, `VIAL_LENGTH = 0.203`).

**What this means for TwinCheck:** our policy *is* the robot program, so the three levels map to modes of one
policy (`cell_policy/`): **baseline** = blind taught program, **watch** = same motion + alerts, **guide** = aim at
what the camera sees and retry. Induced "nudged part" failures = per-rollout nudged layouts in `placement.yaml`
(`cell_policy/make_placement.py`). The legacy-robot (bolt-on Watch) story still holds for the pitch, but on this
cell it is demonstrated as a mode, not on an untouched controller. The ROS 2 stack remains for the twin.

---

## 2. People, contacts and access

| Who | For |
|---|---|
| **Dr. Zeng** (zhenzeng@uwm.edu) | Primary mentor for the FANUC arm, the controller, and the GB10/Spark computers |
| **Dr. Derek Riley** (driley@nvidia.com) | Digital twin questions |
| **Shamar Webster** (webste63@uwm.edu, `#data-and-testbed-requests`) | CSI visits, extra data, supervised physical tests, VPN/account help, Spark help desk in person |
| **Brett Storoe** (storoeb@msoe.edu, `#hackathon-general`) | Website, signup, team, Slack access |
| **Dan Schneider** (danschneider@nvidia.com, `#compute-questions`) | Brev credits and compute |
| Brett Storoe / Dr. Riley / Tanner Cellio | Dataset questions in `#dataset-3-grasping-sorting` |

Slack channels: `#hackathon-announcements`, `#hackathon-general`, `#dataset-3-grasping-sorting`,
`#data-and-testbed-requests`, `#compute-questions`. For a blocking issue: say so in line 1, with what you tried,
the error, and when you need it.

**Team roles (suggested; names to fill):**
| Role | Owner | Does |
|---|---|---|
| Robot and code lead / Spark driver | Rohan | Twin with a coding agent, ROS 2 nodes, comparator, resync, Guide mode |
| Data and testing lead | IST teammate 1 | Labels real photos in CVAT, runs detector training by recipe, trial log, Recorder in sessions |
| Story and coordination lead | IST teammate 2 | Organizers/Slack, CSI bookings, business case, charts, video, one-page summary, finals pitch |

**Compute access:**
- **YAAD GPU Lab:** 8 NVIDIA DGX Sparks at CSI. Team workspace = root + GPU + Docker on one Spark (up to 100 GB
  shared per team), plus batch jobs to any free Spark. Files in `/workspace`. **Everything is deleted after
  the event:** push code daily and copy data, models and DB dumps out.
  - Needs the **CSI VPN** (FortiClient "VPN Only", SSL-VPN, `vpn.csi.uwm.edu`, port 443) for SSH and Sparks.
    The website works without VPN.
  - The CSI account (username `Wicked`) was issued with a **temporary password that must be changed** first:
    at a CSI computer, or over VPN + Remote Desktop to an IP from CSI staff, then Ctrl+Alt+End → Change a password.
    **Never put the password, SSH private keys or tokens in the repo, chat, or docs.**
  - Captain: sign in to YAAD with the @uwm.edu email (type the 6-digit code; don't click the link), generate
    `ssh-keygen -t ed25519`, "Start a team" named Wicked, and share the 8-char team code. Teammates join with their own keys.
  - Login line looks like `ssh yt-<team>@172.30.102.xx`. Guide: https://www.yaadops.com/guide ;
    agent setup: https://www.yaadops.com/agents#setup
  - Use tmux/nohup for long runs. Put a password/token on anything listening on a port (other teams share the Spark).
  - **The Spark is arm64 (Grace).** Use NVIDIA containers for PyTorch/Isaac Sim, not x86 pip wheels.
  - The Sparks can run Isaac Sim and small VLMs during physical tests. Organizers recommend **Cosmos Nano**
    (tested on this exact setup).
- **Brev:** credits (about 150 requested), a backup now that the Spark exists. Always stop instances; log every session.
- **build.nvidia.com:** free model APIs, usable now.
- **MSOE Dell GB10 (Rosie)** and agent tool access: separate messages were pending.
- **CSI Emulate3D workstation:** organizers said it is for datasets 1–2, so we dropped it.
- **Rohan's laptop:** RTX 5060 8 GB (Blackwell, needs PyTorch cu128 wheels). Good for core logic, dashboard,
  calibration and small YOLO training; too small for Isaac Sim (which wants 16 GB).

---

## 3. How the idea evolved (decision log)

1. **Dataset choice.** I first recommended Dataset 2 (fit with Rohan's distributed-tracing experience).
   Rohan preferred **Dataset 3** for resume value: physical AI on a real industrial arm, and it echoes his
   physics-informed ML research (trained on COMSOL simulations because real data was infeasible) and his hackathon
   win (robot failure classification). **Decision: Dataset 3.**
2. **Clarified basics** for Rohan: the digital twin is like a flight simulator; you develop about 90% in the twin and
   do a supervised physical test about 10% at the end; code runs on a PC and talks to the robot through ROS 2
   (pub/sub like Kafka/RabbitMQ, no broker, DDS) and the FANUC driver; only perception is "trained", while grasp
   and motion are regular code plus MoveIt. **The robot doesn't correct itself; our program decides, the controller executes.**
3. **First "winning" plan:** vision-guided pick-and-sort plus a **changeover-time metric** (new object ready from its
   3D model in about 30 min vs days of reteaching).
4. **Novelty pushback** ("don't rebuild existing solutions"). Proposed using the twin at runtime:
   *imagine before acting* (test grasps in sim) and a *shadow twin* (compare expected vs actual).
5. **Prior-art check** (Rohan asked "are you sure it's unique?"). Both ideas exist:
   - Sim-tested grasps: GraspTwin (arXiv 2609.30543), an industrial grasp-generation patent, physics-based grasp
     planning through clutter (RSS 2012, PR2).
   - Sim-vs-real fault detection: Millard et al., IROS 2014 (notes **drift** → periodic reset trade-off);
     Betzer et al. 2024 (cloud twin as runtime watchdog, mobile robot); 2026 apparel case study (runtime
     verification needed because offline twin testing misses floor variance).
   - Commercial: **FANUC iRVision** (taught on the robot per part), **Basler 3D CADMatch** (CAD-based, synthetic
     views, vendor builds a template per part, Basler cameras), **Mujin** (teachless controller with a real-time
     digital twin and self-recovery; replaces the control layer; mainly logistics), Pickit, Cognex, Keyence.
   - **Positioning adopted:** no single piece is new. Our contribution is a specific combination, measured on a real
     industrial arm: a monitor-first, bolt-on shadow twin that checks every pick, **works without controller
     access (legacy robots)**, resyncs after every pick to fight drift (the open problem in Millard 2014),
     plus twin-only training and honest sim-vs-real numbers.
6. **Rockwell's legacy and scalability points** gave the three levels: **Watch** (detect), **Report** (alert
   with expected vs actual), **Guide** (send corrected moves; only with controller access). Watch and Report bolt on
   to any robot without touching its program.
7. **Teammate's proposal** ("Sim-First Grasp and Sort: Multi-View Perception with Finger Co-Design"): a staged
   Baseline → See → Verify → Fit ladder, wrist verify-and-retry, finger co-design, a strong eval plan, a Brev
   budget, and a safety section. Its weaknesses: thin novelty, no legacy angle, fingers as a risky headline, a weak business case.
   **Merged** into the final proposal: teammate's structure plus twin-checked verification, a Watch mode for
   legacy equipment, fingers as a stretch goal, and a business case.
8. **Name:** **TwinCheck**. Pitch: *"Robots today don't know when they've failed. We keep the digital twin
   running alongside the real arm, so the robot knows what should happen, notices when reality disagrees,
   and recovers on its own."*
9. **Teammate's summary** (adopted): first physical version = **monitor the existing program and report**;
   vision-guided moves and retries only if controller access and mentor approval allow; fingers optional later.
10. **Scope:** a simpler "core vs stretch" version was offered when the plan felt hard. On Oct 17, **Rohan chose
    to keep everything ("we are not cutting anything off")**, so the repo implements the full design.

---

## 4. The solution (TwinCheck)

### Plain words
The robot is a blindfolded worker who always reaches for the same spot. We add a supervisor with cameras:
the twin is the **answer key** (what should happen), the cameras read **the student's answers** (what did
happen), and our software marks right or wrong.

### Scenario (proposed, pending team confirmation)
**Original idea (Oct 14):** vials and loose lids sorted into rework/reuse bins. **It does not fit the real
gripper** (12 mm jaws vs a 27 mm vial and a 31.75 mm cap; section 1b). **Updated proposal (Oct 20): small-parts
rework sorting.** Rejected small parts (M6 screws, and 10 mm blocks standing in for machined parts) are dumped on
the table at random positions; the arm sorts them by type into 50 mm bins (`config/twincheck.yaml` `cell.sequence`).
Random positions are why a blind program fails. Vials/caps return only with reviewed custom fingers
(`custom_hardware`); `plan_grasp` refuses them with the stock jaws. Confirm with the team.

### The three checks per pick
| When | Question | Expected from | Observed from | Starting threshold |
|---|---|---|---|---|
| Pre-pick | Is the object where the robot will grab? | Program's pick point (Watch) / planned grasp (Guide) | Table-camera detector (the side camera is near table height, so not used for x/y) | 10 mm or 10° |
| Post-grasp | Is it in the gripper? | Twin: holding | **Wrist camera** (gripper has **no part-present sensor**; organizers confirmed the wrist cam is meant for this) | Any non-holding result fails; unknown = fail |
| Post-place | In the right bin? | Twin: object in target bin | Table-camera detector | Wrong bin, or 15 mm off |

**Resync:** after every pick, set twin object poses to what the cameras see and log the shift (a drift signal).
**Watch:** failures → alerts (urgent on a 2nd in a row). **Guide:** pre-pick miss → re-plan to the observed pose;
failed grasp → retry once from a fresh detection; bad placement or out of retries → stop and alert.
**Baseline mode:** the robot's own program, with checks logged but nobody acting, for the comparison.

### Business case (draft)
Lowest-friction adoption: Watch mode needs only cameras, a PC with a GPU, and a 3D model per part; **no robot
program changes**. Guide is the upgrade once a plant trusts the alerts (needs modern controller options).
Overview cites a NIST estimate of $32–58B/yr in defect costs for US discrete manufacturers and that only
about 3% of manufacturers used AI for inspection (2020 Annual Business Survey). We will measure minutes to prepare a new
object and the share of failures caught before a person would notice.

### Metrics (all vs the blind fixed-motion baseline, twin and real)
Pick success · placement accuracy (mm) · cycle time · **failures caught** (induced: empty grasp, removed mid-grasp,
nudged part) · recovery rate (Guide) · **false alarm rate** (with/without resync) · unseen-object success ·
**changeover time** (3D model → working detector, zero arm time) · sim-to-real gap · finger comparison (stretch).
Difficulty ladder, one variation at a time: fixed pose → random position → random rotation → second object →
light clutter → lighting. Report rates with 95% Wilson intervals; report false alarms and misses separately.

**Sample sizes:** 100 trials per condition in the twin, 20–30 per condition on the real arm (90% from 20 trials is a
CI of about 70–97%, so only claim big real-arm gaps). Induced failures: 100 per type in the twin, at least 10 per type real.
Real-arm time comes in **30-minute slots of about 10 rollouts** each (section 1b): book a dry-run slot around Oct 26–27
and numbers slots around Oct 29 (one slot per mode on the same nudged layouts). A rollout is scored as a whole, while
our database logs per pick, so record per-pick ground truth from the rollout videos with `scripts/ground_truth.py`.

### Data plan
| Dataset | Amount | Source | Use |
|---|---|---|---|
| Twin USD | 1 | Organizers (Dataset 3 page) | Everything in sim |
| Pilot overhead set | 500 | Replicator | Sanity-check labels (draw 20) |
| Main overhead set | 1k → 2k → 4k per object type; stop when the **real** score plateaus | Replicator | Detector |
| Ladder add-ons | about 1k per variation | Replicator | Retrain per rung |
| Wrist set | 1–2k, balanced holding/empty/misaligned | Replicator | Grasp check (if needed) |
| Real overhead test | 100–200 imgs, hand-labeled in CVAT (YOLO OBB export) | CSI visit, or table frames from our rollouts' LeRobotDataset (`run.py --download`) | **Test only** |
| Real wrist test | 30–50 imgs | Same | **Test only** |
| Calibration | Provided: `calibration/cameras.json` in the client package (ArUco-solved, table 0.47 px RMS) | Organizers | `pixel_to_base` in `cell_policy/cell_perception.py` |
| Robot program pick/place points | all | Plant manager | Watch pre-pick check |

Version datasets by name and seed (e.g. `overhead_v3_seed42_n2000`); render once, train many times.

### Models
- **Overhead:** Ultralytics YOLO **OBB** (oriented boxes → center + angle directly), nano first.
  Randomize position/yaw, 1–3 objects plus distractors, lighting, mild textures, camera jitter (about ±1–2 cm, ±1°),
  noise and blur. Handle symmetry (cube 90°, screw 360°, round cap: yaw ignored).
  **Small parts:** a 10 mm cube is about 10 px wide in the full 1920×1080 table image, and only about 3 px at the
  client's default 640×360, so `cell_policy` requests full frames; render and train at high resolution (or tiles).
  **Gate:** recall ≥ 95%, median position ≤ 5 mm, median yaw ≤ 5° on the real test set. Export TensorRT **on the Spark**.
- **Wrist check, escalating:** reference-image difference vs the empty closed gripper → **Cosmos Nano VLM on
  the Spark** (organizer-recommended) → YOLO classifier if needed. **Gate:** ≥ 95% correct and **zero empty-called-held**.

### Decision gates
| Gate | Target date | Pass | If it fails |
|---|---|---|---|
| G1 Controller access | **Answered Oct 20** | The organizers' platform accepts `act(obs)` targets during booked slots (section 1b) | — |
| G2 Crude loop | about Oct 22 | Twin flags a moved object and logs a trial | Simplify the wrist check |
| G3 Detector on real images | about Oct 23–24 | Meets the gate | Widen randomization, re-render once, retrain once; report the gap |
| G4 Dry-run session | about Oct 26–27 | Full pipeline on the real arm; calibration error logged | Use session 2 for both |
| G5 Numbers locked | Oct 29 | All real blocks logged with ground truth | No new features after this |
| G6 Submission | Nov 1 | Video, summary, repo, refs | Does not move |

### Compute budget
About 150 Brev credits requested (render about 8 h, train about 10 h, buffer about 4 h on RTX PRO 6000 / L40S). Now the
**Spark is primary** for rendering, training and live runs; Brev is backup. Log every session; check in at 75 credits.

---

## 5. Organizer feedback on the team plan (Oct 14) and our answers

- Problem aligned; **pick specific items** that fit the fingers or redesign the fingers; explore grasping in the twin.
- Make the **scenario realistic**: how did objects get there, and why sort them? → the rework-sorting story above.
- Much requested data is **already in the provided assets**; explore the twin. **No finger CAD** (measure it).
- Arrange a **CSI visit**; **Dr. Zeng** is the primary contact for the robot and GB10s; twin questions go to **Dr. Riley**.
- build.nvidia.com is usable now; drop the Emulate3D workstation; Brev, MSOE GB10 and agent tools to follow.
- Answers: ER-4iA controller options and read-only state → ask Dr. Zeng. We define sort rules; cubes, vials and lids
  are in the twin. Calibration contents → experiment with the twin. **The twin includes gripper physics, objects, and
  all three cameras, modifiable via agent prompts.** Gripper is driven by an **IO action, with no part-present sensor**.
  Sparks can run Isaac Sim plus small VLMs during tests (try **Cosmos Nano** first).

**Mostly superseded by the platform (section 1b).** Still worth asking: the operator's exact screw placement
convention, custom-finger review lead time, and whether slots allow watching live at CSI.
**Old pending email to Dr. Zeng** (draft exists in the planning chat): ask about FANUC ROS 2 driver controller options
(Remote Motion/RMI, HMI Device, ROS 2 Interface, or External Control Package; the FANUC docs list R-30iB+ with
J519 Stream Motion + R912 Remote Motion or S636), whether the **ER-4iA is supported**, read-only joint
state/program step, a CSI visit, and two supervised sessions. Also needed from him: the MoveIt group name, tool
link, gripper I/O service, workspace limits, and the robot program's pick/place points and waypoints.

---

## 6. Documents produced (claude.ai artifacts)

- Proposal v1 (Shadow Twin): https://claude.ai/code/artifact/3707961b-07b8-4d43-a37e-2001ca101673
- **Merged final proposal** (Sim-First Grasp and Sort with Twin-Checked Picks):
  https://claude.ai/code/artifact/73d79fb1-f4a4-4815-a2b1-358dc6f08383
- **Execution plan** (data, training, setup, ROS design, schema, protocols, gates):
  https://claude.ai/code/artifact/2ae8a3bc-071e-4afb-9d8d-eef0711802c2
- **Easy guide** for IST teammates (plain words, roles, weekly checklists, recipes):
  https://claude.ai/code/artifact/292d2902-b84b-4000-8fb0-b32341d7e5fc
- Team plan form answers (submitted Oct 14); the teammate's original proposal PDF.

---

## 7. The repo

```
CLAUDE.md               this file
config/twincheck.yaml   SINGLE source of truth: objects (symmetry, sizes, target bin), bins, pick points,
                        joint waypoints, thresholds, cameras, model paths, Cosmos endpoint, safety limits
twincheck_core/         pure-Python logic + 37 pytest tests
  geometry.py           Pose2D (mm/deg), symmetry-aware yaw diff, homography, multi-camera fusion
  checks.py             check_pre_pick / check_post_grasp / check_post_place
  comparator.py         per-trial state machine → NONE/ALERT/RETRY/STOP (Watch vs Guide)
  resync.py             match twin objects to detections, report shifts and missing
  timesync.py           StampBuffer: nearest frame to an event within tolerance (like trace span matching)
  grasp.py              grasp across the narrow side; refuses objects wider than the jaws
  stats.py              Wilson CI, threshold sweep, pick threshold (full recall, fewest false alarms)
  config.py             loads config/twincheck.yaml (or $TWINCHECK_CONFIG)
  cell.py               the organizers' cell: UF0 frame, our workspace guard (reach, TCP box, table), interpolation
  pickloop.py           one rollout as act() rows: scan → pre-pick → pick → wrist check → place → post-place,
                        modes baseline | watch | guide, retries, refuses parts that don't fit and unsafe targets
cell_policy/            TwinCheck as the client's policy.py (real arm via the platform; see its README)
  twincheck_policy.py   Policy: perception on request, chunks of ≤100 rows, JSONL + Postgres trial log
  cell_perception.py    table YOLO OBB → table mm via the organizers' pixel_to_base; wrist grasp check
  make_placement.py     client placement.yaml from the config (same or nudged layouts per rollout)
ros2_ws/src/            ROS 2 Jazzy
  twincheck_msgs        DetectedObject, ObjectArray, GraspState, RobotEvent, ExpectedState, CheckResult, ResyncReport
  twincheck_perception  detector_node (YOLO OBB → undistort → homography → fuse cams), grasp_check_node
                        (wrist: reference_diff | classifier | cosmos), infer.py (no-ROS, reused by eval scripts)
  twincheck_checker     robot_state_node (events from joint_states waypoints | manual | guide; run sheet),
                        twincheck_trigger (keyboard p/g/d), expectation_node (config answer key),
                        comparator_node (runs checks, saves images, publishes /check/result),
                        resync_node, alert_node
  twincheck_logger      trial_logger → Postgres
  twincheck_guide       guide_node: MoveItPy pick/place with retry; DISARMED by default, workspace clamp, 10% speed
  twincheck_bringup     twincheck.launch.py (env twin|real, mode baseline|watch|guide, events, wrist_method, armed...)
sim/                    Isaac Sim (run with /isaac-sim/python.sh); sim_config.yaml uses the starter's prim names
  explore_twin.py       prints/saves cameras, joints, rigid bodies + sizes, lights → fill configs
  finger_fit_test.py    narrow side vs jaw opening per object
  replicator_overhead.py  labeled YOLO-OBB images (projects object corners via camera_params)
  replicator_wrist.py   holding/empty/misaligned images + empty_reference.png
  twin_runtime.py       --role trials (scripted picks, induced failures, ground truth to DB, publishes cams/joints)
                        --role shadow (mirrors real joints, applies resync poses, predicts capture at PRE_PICK)
calib/                  make_targets, intrinsics (incl. --webcam), homography (ArUco + robot coords, leave-one-out
                        error, overlay), verify (session-start check), twin_camera_match (K → USD lens; overlay)
perception/             draw_labels, train.py (obb|cls, learning curve, logs to training_runs), eval_real
                        (recall, mm, deg, misses), eval_wrist (confusion, empty-called-held), export_engine, cosmos_probe
scripts/                results.py (all tables + sweep + CSV), ground_truth.py (Recorder CLI), compute_session.py
                        (credit log), simulate_checks.py (no ROS), seed_demo_data.py (fake data for the dashboard)
dashboard/              FastAPI (token on every request; SSE live stream; safe image serving) + React/Vite
db/schema.sql           trials, checks, training_runs, compute_sessions, trial_summary view
docker/ros.Dockerfile, docker-compose.yml (Postgres on localhost + dashboard), Makefile, .github/workflows/ci.yml
docs/                   SETUP.md, WORKFLOW.md (ordered steps + "done when"), SESSION_RUNBOOK.md, ARCHITECTURE.md
```

**Topics:** `/cam/{table,side,wrist}/image_raw`, `/perception/objects`, `/perception/grasp_state`,
`/robot/event`, `/robot/manual_event`, `/twin/expected`, `/twin/objects`, `/twin/set_poses`,
`/twin/resync_report`, `/check/result`, `/alerts`, `/guide/command`.
**Frames/units:** everything in the `table` frame, **mm and degrees**; table x/y = robot_base x/y. The cell's
`tcp_pose`/"tcp" actions are UF0 (same x/y, z − 330 mm); conversions live in `twincheck_core/cell.py`.

### Verified vs not verified (Oct 17)
- ✅ 22 core tests pass; offline simulator runs; pyflakes clean; all files compile.
- ✅ Schema loads in Postgres 16; demo data → `results.py` tables and threshold sweep work.
- ✅ Dashboard API: 401 without a token, summary/trials/checks endpoints work, path traversal blocked; React builds.
- ✅ Homography recovered robot coordinates to under 0.5 mm on a synthetic marker image.
- ✅ (Oct 19–20) **ROS 2 packages build and run** on the laptop (WSL Ubuntu 24.04, Jazzy). Whole pipeline tested
  with `fake_world` standing in for cameras, robot and shadow twin (`scripts/fake_world_run.sh`): Watch caught every
  induced failure with 0 false alarms; Guide (disarmed) retries from fresh detections. Bugs this found and fixed:
  comparator died on its first FAIL (rclpy logger severity); grasp/place event without its pre-pick crashed it;
  post-place passed on parts already in the bin (now counts arrivals vs the pre-pick scene); a late post-place
  was judged against the next trial (trials now kept by id); Guide pre-pick compared to the fixed pick point and
  its retry reset the budget (RobotEvent now carries Guide's target); guide skipped trials if the part wasn't
  visible that instant; lost first event before DDS discovery (fake_world and twin_runtime wait); clock steps
  mixed old/new frames (StampBuffer restarts). Perception nodes were not run (need a model + calibration).
  Oct 20 rerun with the real-cell config (cubes + screws, 50 mm bins): Watch 14/14 failures caught, 0/16 false
  alarms; Guide recovered 3/3 empty grasps and 4/4 nudged parts, 0 false alarms. `fake_world_run.sh` teardown used to
  leave every node running (background jobs ignore SIGINT in a non-interactive shell), so later runs got duplicate
  checks from stale nodes; it now kills the whole process group and refuses to start if old nodes are alive.
- ✅ (Oct 20) **`cell_policy` passes the organizers' mock** (`uv run run.py --mock`: "✓ Ready for the real robot", 0
  rejected actions) in baseline, watch and guide. With no detector/wrist model every check fails safely and Guide
  stops; trials and checks reach Postgres and `results.py` reads them. The mock has no kinematics or real images,
  so no grasp is proven: run the policy's logic in the starter twin before a slot.
- ⚠️ Sim-time risk for the twin run: event stamps come from each node's view of `/clock`; under load a lagging view
  can stamp events early and pull a stale frame, and `sync_gap_ms` won't show it. Watch for this at G2.
- ❌ **Isaac Sim scripts never run**. The starter targets Isaac Sim 6.0 (the ER-4iA asset ships only there); our sim
  scripts target 4.5/5.x API names with fallbacks. Prim names now follow the starter; object origins are unchecked.
- ❌ The ROS Guide node (MoveIt) is untested and no longer the real-arm path; the real arm goes through `cell_policy`.
- All `TODO` values in `config/twincheck.yaml` and `sim/sim_config.yaml` are placeholders, not measurements.

### Known gaps and likely first fixes
- `cell.finger_yaw_offset_deg` (R at which the fingers close along base +x) and `cell.park_tcp_mm` (arm out of the
  table camera's view): measure in the starter twin. Screws need the yaw offset; cubes don't (90° symmetric).
- Pick points and bin positions are our choice now: check reach and camera view in the twin, then regenerate
  placement.yaml. The operator places a screw by its head; we grasp the shank centre.
- A detector for 10 mm cubes and M6 screws (twin renders, high resolution) and a wrist reference image.
- In the trials role, ground truth is written to the latest twin trial row; fine for sequential runs, fragile in parallel.
- Wrist-check ROI (finger gap) must be set per camera; Cosmos model id is TODO.

---

## 8. What to do next (in order)

0. (Oct 20) Team: confirm the small-parts scenario (cubes + screws). Clone the client, run `cell_policy` on the
   mock, fill submission.yaml, and book a dry-run slot (section 1b, `cell_policy/README.md`).
1. Push the repo to the team GitHub repo (ALL AI Network); make CI green (fix any ROS build errors).
2. Everyone: `make test`, `make simulate`. Rohan's laptop tour: simulate → dashboard with demo data →
   YOLO OBB on `dota8.yaml` with **PyTorch cu128** (RTX 5060 Blackwell) → webcam calibration.
3. Captain: create the YAAD team, share the code; change the CSI password; VPN; SSH to the Spark.
4. ~~Email Dr. Zeng about controller access~~ (answered by the platform). Ask the organizers the section 5 questions.
5. On the Spark: download the twin → `sim/explore_twin.py` → fill configs → `sim/finger_fit_test.py`.
6. Follow `docs/WORKFLOW.md`: pilot render → label check → main render → learning curve → wrist data →
   build ROS → twin loop (G2) → CSI calibration + real test sets → eval gates (G3) → ladder + induced failures →
   threshold sweep → Guide in the twin → shadow dry run → real sessions (`docs/SESSION_RUNBOOK.md`) → video.

---

## 9. Rules for the coding agent

- **Safety:** never write code that commands the physical arm by default. The ROS Guide node stays `armed:=false`
  unless a human sets it during a supervised session. On the real cell, targets go only through the organizers'
  client during an approved, operator-supervised slot; keep `twincheck_core/cell.py`'s guard (reach, TCP box,
  table) and the step-size limits, and never loosen them to make a run pass.
- **Never train on real line data.** Real images are test-only.
- **Secrets:** never commit `.env`, keys, passwords or tokens; never print them in logs.
- **One config:** add new tunables to `config/twincheck.yaml`, not hard-coded constants.
- **Logic goes in `twincheck_core`** (pure Python, unit-tested); ROS nodes stay thin wrappers. Add a test for every logic change.
- Units: mm and degrees in the table frame everywhere; convert to meters only at the MoveIt/USD boundary.
- Missed failures cost more than false alarms: unknown states fail, never pass.
- Every number shown to judges must come from the trial database (`scripts/results.py`).
- Verify generated Isaac Sim/ROS code by running it and looking at the output (image, video, topic echo).
- Rohan prefers plain-language explanations with analogies.
