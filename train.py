# train.py - training script with optional rendering of one environment (human mode)
import os
import argparse
from types import SimpleNamespace

from highway_env.envs.highway_env import HighwayEnv
from highway_env.wrappers.tuple_flatten import TupleFlattenWrapper

# Stable Baselines 3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
except Exception as e:
    raise ImportError(
        "Please install stable-baselines3 to run training: pip install stable-baselines3"
    ) from e


class RenderCallback(BaseCallback):
    """
    Callback that renders the first environment in the vector env every render_freq calls.
    Uses render(mode='human') so pygame window is opened when supported.
    """

    def __init__(self, render_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = max(1, int(render_freq))

    def _get_first_env(self):
        # self.training_env is a VecEnv (DummyVecEnv). .envs is the list of env instances.
        env = None
        try:
            env = self.training_env.envs[0]
        except Exception:
            # fallback: try attribute
            env = getattr(self.training_env, "env", None)
        # If wrapper (like VecEnv or wrappers) hold inner env under .env or .unwrapped
        # Try to find the underlying env that implements render
        try:
            # Unwrap common wrappers
            if hasattr(env, "env"):  # some wrappers store env in .env
                env = env.env
        except Exception:
            pass
        return env

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq != 0:
            return True
        env = self._get_first_env()
        if env is None:
            return True
        # Ensure env has a spec so AbstractEnv.render won't assert
        if getattr(env, "spec", None) is None:
            env.spec = SimpleNamespace(id="HighwayEnvLocal-v0")
        # Try to render in human mode (open pygame window); fallback to rgb_array
        try:
            env.render(mode="human")
        except TypeError:
            # Some implementations accept no-arg render()
            try:
                env.render()
            except Exception:
                # fallback to rgb_array (no window) - ignore
                try:
                    _ = env.render(mode="rgb_array")
                except Exception:
                    pass
        except Exception:
            # ignore render errors during training
            pass
        return True


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
    # Flatten the tuple observation for SB3
    env = TupleFlattenWrapper(env)
    # Ensure spec exists so render won't assert
    if getattr(env, "spec", None) is None:
        env.spec = SimpleNamespace(id="HighwayEnvLocal-v0")
    return env


def train(total_timesteps: int = 100_000, save_path: str = "ppo_highway", render: bool = False, render_freq: int = 1):
    # Single env so we can render it
    def _init():
        return make_env()

    env = DummyVecEnv([_init])
    model = PPO("MlpPolicy", env, verbose=1)

    cb = RenderCallback(render_freq=render_freq, verbose=1) if render else None

    model.learn(total_timesteps=total_timesteps, callback=cb)
    os.makedirs(save_path, exist_ok=True)
    model.save(os.path.join(save_path, "model"))
    print("Saved model to", save_path)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of training timesteps")
    parser.add_argument("--save", type=str, default="ppo_highway", help="Directory to save model")
    parser.add_argument("--render", action="store_true", help="Render the environment during training (shows window)")
    parser.add_argument("--render_freq", type=int, default=1, help="Render every N callback calls (lower -> more frequent)")
    args = parser.parse_args()

    # Ensure OFFSCREEN_RENDERING is not set for this process when we want to show the window
    if args.render and os.environ.get("OFFSCREEN_RENDERING") == "1":
        os.environ.pop("OFFSCREEN_RENDERING", None)

    train(total_timesteps=args.timesteps, save_path=args.save, render=args.render, render_freq=args.render_freq)