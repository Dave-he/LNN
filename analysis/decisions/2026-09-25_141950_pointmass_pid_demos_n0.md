# PointMass PID expert demos (n_ped=0) — 2026-09-25_141950

Phase B of PRD-MSD-LNN. PD controller rollouts on ``PointMassNavLite``.

* n_episodes: **500**
* n_transitions: **16000**
* reach_rate: **1.000** (500/500)
* collision_rate: **0.000** (0/500)
* avg_return: **8.187**
* avg_steps: **32.00**

Controller gains:
* P_v=0.1, P_w=2.5, D_w=0.6
* v_max=0.1, w_max=1.571
* obstacle_push=0.06

Stored as ``obs / actions / next_obs`` flat tensors in the JSON companion.
Use ``scripts/experiment_sncp_ppo_lite.py --il-warmstart`` to load and train.
