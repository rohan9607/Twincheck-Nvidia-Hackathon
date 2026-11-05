# TwinCheck demo: script and cheat sheet

Poster/demo session, MSOE atrium. You have **2–3 minutes** to cover the problem, the solution running on the CSI hardware,
and the results, then take questions. Judges score: a grounded manufacturing problem · effectiveness and validation ·
physical operation and reliability · engineering practicality · value and generalizability · broader impacts.

## Setup (do this before the round)

1. **Laptop:** charged, HDMI adapter, notifications off, display set to the TV's resolution (1920×1080 works best).
2. **Demo page:** open `demo/index.html` in Chrome or Edge, press **F11** (full screen), and start on view **1**.
   It runs offline and needs only this repo folder.
3. **Live detector:** in a terminal, run `conda activate twincheck` and then `python demo/live_detect.py`. Run it once
   before the round so the GPU is warm, then keep its window behind the browser.
   The detector needs files that aren't in git: the model weights (`models/overhead_v4_n3000.pt`), the organizers' client
   package next to the repo (for its camera calibration), and the `twincheck` conda env. All three are on Rohan's laptop.
   On any other laptop, open the saved output in `demo/assets/live/` instead (same images, already annotated).
4. **Submitted video:** download it from YouTube beforehand and keep it in a tab or player. Don't rely on the venue wifi.
5. **Between visitors:** leave view **1** running. Its Guide clip loops, so the screen is never idle.

Keys on the page: **1–6** switch views · **Space** play/pause · **← →** change rollout · **R** restart · **F** full screen.
In the live detector: **Space** next image · **A** auto-play · **F** full screen · **Q** quit.

## The 2½-minute walkthrough

**[View 1 · Overview] 0:00–0:25: the problem**
> "Factories teach a robot fixed spots and replay them forever. On a high-mix line, parts never land exactly there.
> On the real FANUC cell, hand-placed parts sat 5 to 33 mm off and bins 1 to 3 cm off, but these jaws only leave
> about 3 mm of room. So a taught program grabs air or crashes, and nobody notices. That's the problem we solved."

**[View 2 · Side by side, rollout 1, press Space] 0:25–1:15: the solution running on CSI hardware**
> "This is the real cell's own recording: the same layout, the blind taught program on top and TwinCheck below."

Pause on these moments (watch the **rollout time** counter):
- **~12 s:** top bar turns red: *"cube is off the pick point by 23 mm; this pick will likely miss."*
  > "Even running beside the blind program, TwinCheck predicted the miss before the arm moved."
- **~30 s:** blind: *"gripper is empty."* TwinCheck: *"object held."*
  > "TwinCheck aims at where the camera sees the part and turns the fingers to its angle."
- **~45–50 s:** the blind arm comes down on the screw and the controller faults. TwinCheck has placed the cube in the
  real bin, which is 39 mm from where it was planned.
  > "Look, act, then check three times: is the part there, is it in the hand, is it in the right bin."

**[View 3 · Sim → Real, or switch to the live detector window] 1:15–1:45: physical AI**
> "The AI never saw a real training image. We rendered 3,000 images of this exact cell in NVIDIA Isaac Sim, with
> labels computed from the 3D scene, and trained a 5.7 MB YOLO model on a DGX Spark in under two hours. Here it is live
> on real cell frames, in about 20 ms per frame: the red X is where a taught program would reach, and the box is
> where the part really is."

**[View 1, the score card] 1:45–2:15: results**
> "The organizers' operator scored both runs on the same 10 layouts. The blind program sorted 0 of 10, with 7 controller
> faults. TwinCheck sorted 9 of 10, with 1 fault and no interventions. The 95% intervals don't overlap. And every
> change was tested in the twin first, which went 20 for 20 before we spent robot time."

**[View 5 → 6] 2:15–2:40: why it matters, the limits, then hand over**
> "For a plant, a new part means a new 3D model, not days of re-teaching: about two hours of compute and zero robot
> hours. Watch mode bolts onto an existing program, so it fits the decades-old robots Rockwell told us about. The limits:
> 10 rollouts per condition, a speed-capped arm, and only parts that fit 12 mm jaws. Happy to take questions, or show
> you any rollout."

## Numbers to have ready (all official or from our logs)

| | Blind taught program | TwinCheck Guide |
| --- | --- | --- |
| Rollouts fully sorted | 0 / 10 | **9 / 10** |
| 95% Wilson interval | 0–28% | 60–98% |
| Controller faults | 7 | 1 |
| Interventions | 0 | 0 |
| Time per 3-part sort | – | 96.5 s |

- Fisher's exact test, 9/10 vs 0/10: **p ≈ 0.0001** (two-sided).
- Watch flagged **20/20** of the blind program's picks as "will likely miss" before the arm moved; **19** truly missed.
- Twin test bench: moved-bin test **2/5 → 5/5** after teaching the model to see bins, then a **20/20** sort test.
- Detector: YOLO11n-OBB, 2.66 M parameters, 5.7 MB, 3 classes (cube, screw, bin); 3,000 renders (~12,000 labels),
  1280 px, 60 epochs, ~1.7 h on a DGX Spark; ~20 ms per frame on the laptop's RTX 5060.
- Simulator exams (100 images each): 99.4–100% found, typical error 0.2–0.5 mm.
- Runs: TwinCheck `r20261031-79333b`, baseline `r20261030-dcd07a` (Oct 30). 7 real runs, 73 recorded rollouts in total.

## Likely questions

**"Why no real images in training?"** It's the hackathon rule, and it's how a high-mix line has to work: a new product has no
real photos yet. Labels from the twin are free and exact. Real frames were used only to test.

**"How do pixels become millimetres?"** The organizers' camera calibration. Each part's pixel ray is cut at the height of the
part's top face, and at the bin floor if the part sits inside a bin. Bins are cut at their rim height.

**"Why YOLO and not a VLM like Cosmos?"** We need each part's angle to turn the fingers, plus millimetre accuracy at 10 Hz.
An oriented-box detector gives centre + angle in ~20 ms. A VLM is a candidate for the "is it in the hand?" check.

**"What failed in your 9/10?"** Rollout 8: the screw landed on the edge of its bin (our check flagged it), and then the
controller faulted on the third pick. We haven't confirmed the root cause.

**"Any false alarms?"** Yes, 2. In rollouts 6 and 7 our post-place check said "screw in the wrong bin", but the operator scored
both as fully sorted. Bins seen at an angle are still hard for that check. We report misses and false alarms separately.

**"What was the sim-to-real gap?"** Three things the twin didn't model at first:
- The real arm lags about 0.8 s and moves at about 55 mm/s, so it cut corners. Fix: wait for measured arrival before every step.
- The real cubes are paler than the simulated ones. Fix: colour variation in the renders.
- The wrist wound its air hoses toward its limit, half a turn per pick. Fix: a grasp-angle band and a 270° wrist guard.

Each fix was reproduced and tested in the twin first.

**"Does Watch really need no controller access?"** Watch only needs the camera and the robot's pick points. It adds no motion.
On this cell our policy *is* the robot program (the organizers' platform), so we demonstrated Watch as a mode that keeps the
blind program's exact motion and only adds alerts.

**"How fast can you add a new part?"** For a new class: its 3D model, ~15 min of rendering, ~1.7 h of training, zero robot
time. That's how we added bins as a class.

**"Is it safe?"** Every target passes guards before it's sent: reach, a fingertip box, the table surface, and the wrist-turn
limit. The cell enforces its own limits as well, and every real run was operator-supervised.

**"Why is it slow (96.5 s)?"** The cell's arm is speed-capped at about 55 mm/s, and we wait for arrival before closing the
jaws. We traded speed for reliability.

**"Is 10 rollouts enough?"** It's a small sample, so we report intervals: 60–98% vs 0–28%, which don't overlap, and Fisher's
exact test gives p ≈ 0.0001. More rollouts would tighten them.

**"What would it cost a plant?"** An overhead camera, a PC with a GPU, and a 3D model per part. Start with Watch (alerts only),
then switch on Guide where the controller accepts external targets.

**"What's next?"** A wrist-camera grasp check trained in the twin, more real rollouts, custom fingers for vials and caps, and a
live operator dashboard for Watch alerts.
