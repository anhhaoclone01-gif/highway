# quick_test.py
import os

# Avoid opening pygame windows
os.environ["OFFSCREEN_RENDERING"] = "1"

import numpy as np
from highway_env.envs.highway_env import HighwayEnv

# Use full import path for wrapper if you need it (not used here)
# from highway_env.wrappers.tuple_flatten import TupleFlattenWrapper

def make_env():
    env = HighwayEnv()
    env.configure(
        {
            # Use Kinematics only to avoid observation_factory errors in current code
            "observation": {"type": "Kinematics", "vehicles_count": 6, "normalize": True},
            "action": {"type": "DiscreteMetaAction", "longitudinal": False, "lateral": True},
            # enable safety soft penalties so _reward uses safety term
            "safety_weight": 1.0,
            "lane_change_danger_weight": 1.0,
        }
    )
    env.define_spaces()
    return env

def danger_metrics_to_vector(danger: dict, env: HighwayEnv):
    """
    Convert danger dict from env._compute_danger_metrics() into normalized 6-d vector:
    [ttc_norm, gap_norm, rel_v_norm, a_req_norm, danger_flag, min_ttc_norm]
    Uses same normalizations as our planned DangerObservation.
    """
    max_ttc = 10.0
    max_gap = 100.0
    max_rel_speed = getattr(env.vehicle.__class__, "MAX_SPEED", 40.0)
    max_a_req = 10.0

    def norm_ttc(v):
        if v == float("inf"):
            return 1.0
        return float(np.clip(v / max_ttc, 0.0, 1.0))

    ttc_norm = norm_ttc(danger.get("ttc_front", float("inf")))
    gap_norm = float(np.clip(danger.get("gap_front", 0.0) / max_gap, 0.0, 1.0))
    rel_v = danger.get("rel_v_front", 0.0)
    rel_v_norm = float(np.clip((rel_v + max_rel_speed) / (2 * max_rel_speed), 0.0, 1.0))
    a_req_norm = float(np.clip(danger.get("a_req_front", 0.0) / max_a_req, 0.0, 1.0))
    danger_flag = 1.0 if danger.get("danger_flag", False) else 0.0
    min_ttc_norm = norm_ttc(danger.get("min_ttc", float("inf")))
    return np.array([ttc_norm, gap_norm, rel_v_norm, a_req_norm, danger_flag, min_ttc_norm], dtype=np.float32)

if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()
    print("obs type:", type(obs))
    print("Initial info keys:", list(info.keys()))
    # Get danger metrics via env helper
    danger = env._compute_danger_metrics() if hasattr(env, "_compute_danger_metrics") else None
    print("Initial danger dict:", danger)
    if danger is not None:
        danger_vec = danger_metrics_to_vector(danger, env)
        print("Mapped danger vector:", danger_vec)

    # Step a few times and print reward and danger info
    for i in range(8):
        a = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(a)
        danger = info.get("danger") or (env._compute_danger_metrics() if hasattr(env, "_compute_danger_metrics") else None)
        if danger is not None:
            danger_vec = danger_metrics_to_vector(danger, env)
        else:
            danger_vec = None
        print(f"step {i}: reward={rew:.4f}, danger_flag={danger.get('danger_flag') if danger else None}, danger_vec={danger_vec}")
        if term or trunc:
            print("Terminated or truncated at step", i)
            break
    env.close()