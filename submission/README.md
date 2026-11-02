# TwinCheck: submission materials (Team Wicked)

| File | What it is |
| --- | --- |
| `TwinCheck_Summary.pdf` | Project summary: problem, physical AI approach, data, use of the CSI cell, evidence |
| `TwinCheck_Wicked_presentation.pptx` | Finals presentation (open in desktop PowerPoint to play the embedded videos) |
| `system_architecture.png` | System architecture: trained in the twin (offline), run on the real cell (online) |
| `evidence/evidence_baseline_r1.mp4` | Blind baseline, rollout 1 (run r20261030-dcd07a): top, wrist and side cameras, synchronized, with TwinCheck's live log |
| `evidence/evidence_guide_r1.mp4` | TwinCheck Guide, rollout 1 (run r20261031-79333b), same layout (played at 2x) |
| `evidence/layout_*.png` | Rollout 1 layout: taught spots vs measured part and bin positions |
| `logs/baseline.log`, `logs/guide.log` | Every check of every real rollout, the official end reason per rollout, and the client logs |

## Official results (organizers' scores, Oct 30, 2026; same 10 layouts)

| Measure | Blind taught program | TwinCheck Guide |
| --- | --- | --- |
| Rollouts fully sorted | 0 / 10 | **9 / 10** |
| 95% confidence interval (Wilson) | 0-28% | **60-98%** |
| Controller faults | 7 | **1** |
| Operator interventions | 0 | 0 |
| Average time per 3-part sort | - | 96.5 s |

Runs on hackathon.sidclab.org: TwinCheck `r20261031-79333b`, baseline `r20261030-dcd07a`.
Real data was used only for testing and analysis; all training data came from Isaac Sim.
