# evaluate.py
import os
import numpy as np
from stable_baselines3 import PPO

from highway_env.envs.highway_env import HighwayEnv
from highway_env.wrappers.tuple_flatten import TupleFlattenWrapper

def make_env():
    env = HighwayEnv()
    env.configure(
        {
            "observation": {
                "type": "TupleObservation",
                "observation_configs": [
                    {"type": "Kinematics", "vehicles_count": 6, "normalize": True},
                    {"type": "DangerObservation", "max_ttc": 10.0, "max_gap": 100.0, "t_safe": 2.0, "a_max_threshold": 6.0},
                ],
            },
            "action": {"type": "DiscreteMetaAction", "longitudinal": False, "lateral": True},
            "safety_weight": 1.0,
            "lane_change_danger_weight": 1.0,
        }
    )
    env.define_spaces()
    env = TupleFlattenWrapper(env)
    return env

def evaluate(model_path, n_episodes=20):
    env = make_env()
    model = PPO.load(model_path)
    stats = {"collisions": 0, "steps": 0, "total_speed": 0.0, "danger_count": 0, "episodes": 0}
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_collided = False
        ep_danger_count = 0
        ep_steps = 0
        ep_speed_sum = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1
            ep_speed_sum += env.env.vehicle.speed if hasattr(env, "env") else env.vehicle.speed
            danger = info.get("danger")
            if danger and danger.get("danger_flag"):
                ep_danger_count += 1
            if info.get("crashed") or info.get("rewards", {}).get("collision_reward", 0) != 0:
                ep_collided = True
            if terminated or truncated:
                break
        stats["collisions"] += int(ep_collided)
        stats["steps"] += ep_steps
        stats["total_speed"] += ep_speed_sum
        stats["danger_count"] += ep_danger_count
        stats["episodes"] += 1
    env.close()
    print("Episodes:", stats["episodes"])
    print("Collision rate:", stats["collisions"] / stats["episodes"])
    print("Avg speed:", stats["total_speed"] / stats["steps"])
    print("Avg danger fraction per episode:", stats["danger_count"] / stats["episodes"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to SB3 model (.zip)")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.model, n_episodes=args.episodes)