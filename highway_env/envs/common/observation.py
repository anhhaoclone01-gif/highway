# observation.py
from __future__ import annotations

from collections import OrderedDict
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.lane import AbstractLane
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle




if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class ObservationType:
    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class DangerObservation(ObservationType):
    """
    Observation that computes danger metrics:
      - TTC to front vehicle
      - gap to front vehicle (m)
      - relative speed to front vehicle (m/s)
      - required deceleration to avoid collision (a_req, m/s^2)
      - danger flag if TTC < t_safe or a_req > a_max_threshold
      - min TTC among neighbors

    Returns normalized 6-dim vector in [0,1]:
    [ttc_norm, gap_norm, rel_v_norm, a_req_norm, danger_flag, min_ttc_norm]
    """

    def __init__(
        self,
        env,
        max_ttc: float = 10.0,
        max_gap: float = 100.0,
        max_rel_speed: float | None = None,
        max_a_req: float = 10.0,
        t_safe: float = 2.0,
        a_max_threshold: float = 6.0,
        include_rear: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(env)
        self.max_ttc = float(max_ttc)
        self.max_gap = float(max_gap)
        self.max_rel_speed = float(max_rel_speed) if max_rel_speed is not None else float(
            getattr(__import__("highway_env.vehicle.kinematics", fromlist=["Vehicle"]).Vehicle, "MAX_SPEED", 40.0)
        )
        self.max_a_req = float(max_a_req)
        self.t_safe = float(t_safe)
        self.a_max_threshold = float(a_max_threshold)
        self.include_rear = include_rear

    def space(self):
        from gymnasium import spaces
        import numpy as np
        return spaces.Box(shape=(6,), low=0.0, high=1.0, dtype=np.float32)

    @staticmethod
    def _longitudinal_distance(ego, other) -> float:
        import numpy as np
        delta = other.position - ego.position
        heading = np.array([np.cos(ego.heading), np.sin(ego.heading)])
        return float(np.dot(delta, heading))

    @staticmethod
    def _longitudinal_rel_speed(ego, other) -> float:
        import numpy as np
        heading = np.array([np.cos(ego.heading), np.sin(ego.heading)])
        return float(np.dot(ego.velocity - other.velocity, heading))

    @staticmethod
    def _compute_ttc_and_a_req(gap: float, rel_v: float, eps: float = 1e-6):
        INF = float("inf")
        if gap <= 0:
            return 0.0, float("inf")
        if rel_v <= 0:
            return INF, 0.0
        ttc = gap / rel_v
        a_req = (rel_v ** 2) / (2.0 * gap) if gap > eps else float("inf")
        return float(ttc), float(a_req)

    def observe(self):
        import numpy as np
        if not self.env.road or not self.observer_vehicle:
            return np.zeros((6,), dtype=np.float32)

        ego = self.observer_vehicle

        ttc_front = float("inf")
        a_req_front = 0.0
        gap_front = float("inf")
        rel_v_front = 0.0
        min_ttc = float("inf")

        for other in self.env.road.vehicles:
            if other is ego or not getattr(other, "solid", True):
                continue

            longitudinal = self._longitudinal_distance(ego, other)
            rel_v = self._longitudinal_rel_speed(ego, other)

            if longitudinal > 0:
                gap = longitudinal - (other.LENGTH + ego.LENGTH) / 2.0
                ttc, a_req = self._compute_ttc_and_a_req(gap, rel_v)
                if ttc < ttc_front:
                    ttc_front = ttc
                    a_req_front = a_req
                    gap_front = max(gap, 0.0)
                    rel_v_front = rel_v
                if rel_v > 0 and gap > 0:
                    ttc_neighbor = gap / rel_v
                    if 0.0 <= ttc_neighbor < min_ttc:
                        min_ttc = ttc_neighbor
            elif self.include_rear and longitudinal < 0:
                gap = -longitudinal - (other.LENGTH + ego.LENGTH) / 2.0
                rel_v_rear = -rel_v
                ttc_rear, a_req_rear = self._compute_ttc_and_a_req(gap, rel_v_rear)
                if ttc_rear < min_ttc:
                    min_ttc = ttc_rear

        def norm_ttc(ttc_val):
            if ttc_val == float("inf"):
                return 1.0
            return float(np.clip(ttc_val / self.max_ttc, 0.0, 1.0))

        ttc_norm = norm_ttc(ttc_front)
        gap_norm = float(np.clip(gap_front / self.max_gap, 0.0, 1.0))
        rel_v_norm = float(
            np.clip((rel_v_front + self.max_rel_speed) / (2 * self.max_rel_speed), 0.0, 1.0)
        )
        a_req_norm = float(np.clip(a_req_front / self.max_a_req, 0.0, 1.0))
        min_ttc_norm = norm_ttc(min_ttc)

        danger_flag = 0.0
        if (ttc_front < self.t_safe) or (a_req_front > self.a_max_threshold):
            danger_flag = 1.0

        obs = np.array(
            [ttc_norm, gap_norm, rel_v_norm, a_req_norm, danger_flag, min_ttc_norm],
            dtype=np.float32,
        )
        return obs


class GrayscaleObservation(ObservationType):
    """
    An observation class that collects directly what the simulator renders.

    Also stacks the collected frames as in the nature DQN.
    The observation shape is C x W x H.

    Specific keys are expected in the configuration dictionary passed.
    Example of observation dictionary in the environment config:
        observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84)
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion,
        }
    """

    def __init__(
        self,
        env: AbstractEnv,
        observation_shape: tuple[int, int],
        stack_size: int,
        weights: list[float],
        scaling: float | None = None,
        centering_position: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(env)
        self.observation_shape = observation_shape
        self.shape = (stack_size,) + self.observation_shape
        self.weights = weights
        self.obs = np.zeros(self.shape, dtype=np.uint8)

        # The viewer configuration can be different between this observation and env.render() (typically smaller)
        viewer_config = env.config.copy()
        viewer_config.update(
            {
                "offscreen_rendering": True,
                "screen_width": self.observation_shape[0],
                "screen_height": self.observation_shape[1],
                "scaling": scaling or viewer_config["scaling"],
                "centering_position": centering_position
                or viewer_config["centering_position"],
            }
        )
        self.viewer = EnvViewer(env, config=viewer_config)

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)

    def observe(self) -> np.ndarray:
        new_obs = self._render_to_grayscale()
        self.obs = np.roll(self.obs, -1, axis=0)
        self.obs[-1, :, :] = new_obs
        return self.obs

    def _render_to_grayscale(self) -> np.ndarray:
        self.viewer.observer_vehicle = self.observer_vehicle
        self.viewer.display()
        raw_rgb = self.viewer.get_image()  # H x W x C
        raw_rgb = np.moveaxis(raw_rgb, 0, 1)
        return np.dot(raw_rgb[..., :3], self.weights).clip(0, 255).astype(np.uint8)


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: AbstractEnv, horizon: int = 10, **kwargs: dict) -> None:
        super().__init__(env)
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(
                shape=self.observe().shape, low=0, high=1, dtype=np.float32
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(
                (3, 3, int(self.horizon * self.env.config["policy_frequency"]))
            )
        grid = compute_ttc_grid(
            self.env,
            vehicle=self.observer_vehicle,
            time_quantization=1 / self.env.config["policy_frequency"],
            horizon=self.horizon,
        )
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0 : lf + 1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0 : vf + 1, :, :]
        return clamped_grid.astype(np.float32)


class KinematicObservation(ObservationType):
    """Observe the kinematics of nearby vehicles."""

    FEATURES: list[str] = ["presence", "x", "y", "vx", "vy"]

    def __init__(
        self,
        env: AbstractEnv,
        features: list[str] = None,
        vehicles_count: int = 5,
        features_range: dict[str, list[float]] = None,
        absolute: bool = False,
        order: str = "sorted",
        normalize: bool = True,
        clip: bool = True,
        see_behind: bool = False,
        observe_intentions: bool = False,
        include_obstacles: bool = True,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.include_obstacles = include_obstacles

    def space(self) -> spaces.Space:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features)),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(
                self.observer_vehicle.lane_index
            )
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [
                    -AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                    AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                ],
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])
        # Add nearby traffic
        close_vehicles = self.env.road.close_objects_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
            sort=self.order == "sorted",
            vehicles_only=not self.include_obstacles,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            vehicles_df = pd.DataFrame.from_records(
                [
                    v.to_dict(origin, observe_intentions=self.observe_intentions)
                    for v in close_vehicles[-self.vehicles_count + 1 :]
                ]
            )
            df = pd.concat([df, vehicles_df], ignore_index=True)

        df = df[self.features]

        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)











class TupleObservation(ObservationType):
    def __init__(
        self, env: AbstractEnv, observation_configs: list[dict], **kwargs
    ) -> None:
        super().__init__(env)
        self.observation_types = [
            observation_factory(self.env, obs_config)
            for obs_config in observation_configs
        ]

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.observation_types)


class ExitObservation(KinematicObservation):
    """Specific to exit_env, observe the distance to the next exit lane as part of a KinematicObservation."""

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        ego_dict = self.observer_vehicle.to_dict()
        exit_lane = self.env.road.network.get_lane(("1", "2", -1))
        ego_dict["x"] = exit_lane.local_coordinates(self.observer_vehicle.position)[0]
        df = pd.DataFrame.from_records([ego_dict])[self.features]

        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = pd.concat(
                [
                    df,
                    pd.DataFrame.from_records(
                        [
                            v.to_dict(
                                origin, observe_intentions=self.observe_intentions
                            )
                            for v in close_vehicles[-self.vehicles_count + 1 :]
                        ]
                    )[self.features],
                ],
                ignore_index=True,
            )
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)








def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    """
    Factory that returns an ObservationType instance according to the config.

    Ensures the variable 't' is defined from the config and maps names to classes.
    """
    # Ensure we always extract the requested type string
    t = config.get("type", "")

    # Existing observation types (keep the ones already implemented in your file)
    # Add / adapt entries as needed to match classes defined in this module or siblings.
    if t in ("TimeToCollision", "TimeToCollisionObservation"):
        return TimeToCollisionObservation(env, **config)
    elif t in ("Kinematics", "KinematicObservation"):
        return KinematicObservation(env, **config)
    elif t in ("GrayscaleObservation", "Grayscale"):
        return GrayscaleObservation(env, **config)
    elif t in ("TupleObservation", "Tuple"):
        # Ensure you have a TupleObservation class implemented in this module
        return TupleObservation(env, **config)
    elif t in ("ExitObservation", "Exit"):
        return ExitObservation(env, **config)
    elif t in ("DangerObservation", "Danger"):
        return DangerObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")