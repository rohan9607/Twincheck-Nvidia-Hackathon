# TwinCheck: submission materials (Team Wicked)

| File | What it is |
| --- | --- |
| [`TwinCheck_Summary.pdf`](TwinCheck_Summary.pdf) | Project summary: problem, physical AI approach, data, use of the CSI cell, evidence |
| [`TwinCheck_Wicked_presentation.pptx`](TwinCheck_Wicked_presentation.pptx) | Finals presentation (open in desktop PowerPoint to play the embedded videos) |
| [`system_architecture.png`](system_architecture.png) | System architecture: trained in the twin (offline), run on the real cell (online) |
| [`rollouts/baseline/`](rollouts/baseline/) | All 10 rollouts of the blind baseline (run r20261030-dcd07a), one clip each |
| [`rollouts/guide/`](rollouts/guide/) | All 10 rollouts of TwinCheck Guide (run r20261031-79333b), one clip each, played at 2x |
| [`evidence/evidence_baseline_r1.mp4`](evidence/evidence_baseline_r1.mp4) | Blind baseline, rollout 1, at higher resolution |
| [`evidence/evidence_guide_r1.mp4`](evidence/evidence_guide_r1.mp4) | TwinCheck Guide, rollout 1, same layout, at higher resolution (played at 2x) |
| [`evidence/layout_baseline.png`](evidence/layout_baseline.png), [`evidence/layout_guide.png`](evidence/layout_guide.png) | Rollout 1 layout: taught spots vs measured part and bin positions |
| [`logs/baseline.log`](logs/baseline.log), [`logs/guide.log`](logs/guide.log) | Every check of every real rollout, the official end reason per rollout, and the client logs |

Every clip is cut from the cell's own recording (top, wrist and side cameras in sync), with the official score at the top
and TwinCheck's live log at the bottom.

## Official results (organizers' scores, Oct 30, 2026; same 10 layouts)

| Measure | Blind taught program | TwinCheck Guide |
| --- | --- | --- |
| Rollouts fully sorted | 0 / 10 | **9 / 10** |
| 95% confidence interval (Wilson) | 0-28% | **60-98%** |
| Controller faults | 7 | **1** |
| Operator interventions | 0 | 0 |
| Average time per 3-part sort | - | 96.5 s |

## Every rollout, baseline vs Guide

| Rollout | Blind baseline (`r20261030-dcd07a`) | Watch flags | Video | TwinCheck Guide (`r20261031-79333b`) | Checks passed | Video |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Fail (controller fault), 50 s | 4 of 4 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_01.mp4) | **Success**, 109 s | 9 of 9 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_01.mp4) |
| 2 | Fail (controller fault), 78 s | 7 of 7 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_02.mp4) | **Success**, 92 s | 8 of 8 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_02.mp4) |
| 3 | Fail (controller fault), 14 s | 1 of 1 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_03.mp4) | **Success**, 102 s | 9 of 9 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_03.mp4) |
| 4 | Fail, 93 s | 7 of 9 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_04.mp4) | **Success**, 100 s | 9 of 9 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_04.mp4) |
| 5 | Fail (controller fault), 14 s | 1 of 1 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_05.mp4) | **Success**, 100 s | 9 of 9 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_05.mp4) |
| 6 | Fail, 96 s | 9 of 9 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_06.mp4) | **Success**, 89 s | 7 of 8 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_06.mp4) |
| 7 | Fail, 97 s | 9 of 9 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_07.mp4) | **Success**, 88 s | 7 of 8 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_07.mp4) |
| 8 | Fail (controller fault), 43 s | 4 of 4 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_08.mp4) | Fail (controller fault), 76 s | 6 of 7 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_08.mp4) |
| 9 | Fail (controller fault), 13 s | 1 of 1 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_09.mp4) | **Success**, 94 s | 9 of 9 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_09.mp4) |
| 10 | Fail (controller fault), 14 s | 1 of 1 checks failed | [clip](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/baseline/rollout_10.mp4) | **Success**, 95 s | 8 of 8 | [clip (2x)](https://github.com/rohan9607/Twincheck-Nvidia-Hackathon/blob/main/submission/rollouts/guide/rollout_10.mp4) |

Runs on hackathon.sidclab.org: TwinCheck `r20261031-79333b`, baseline `r20261030-dcd07a`.
Real data was used only for testing and analysis; all training data came from Isaac Sim.

In Guide rollouts 6 and 7, our post-place check reported "screw in the wrong bin", but the operator scored both as fully
sorted, so those were false alarms. In rollout 8 the check flagged the screw on the edge of its bin, and the controller
then faulted on the third pick.
