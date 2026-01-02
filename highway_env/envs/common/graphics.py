from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

import numpy as np
import pygame

from highway_env.envs.common.action import (
    ActionType,
    ContinuousAction,
    DiscreteMetaAction,
)
from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.vehicle.graphics import VehicleGraphics


if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv
    from highway_env.envs.common.abstract import Action


class EnvViewer:
    """A viewer to render a highway driving environment."""

    SAVE_IMAGES = False
    agent_display = None

    def __init__(self, env: AbstractEnv, config: dict | None = None) -> None:
        self.env = env
        self.config = config or env.config
        self.offscreen = self.config["offscreen_rendering"]
        self.observer_vehicle = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0
        self.directory = None

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.config["screen_width"], self.config["screen_height"])

        # A display is not mandatory to draw things. Ignoring the display.set_mode()
        # instruction allows the drawing to be done on surfaces without
        # handling a screen display, useful for e.g. cloud computing
        if not self.offscreen:
            self.screen = pygame.display.set_mode(
                [self.config["screen_width"], self.config["screen_height"]]
            )
        if self.agent_display:
            self.extend_display()
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = self.config.get(
            "scaling", self.sim_surface.INITIAL_SCALING
        )
        self.sim_surface.centering_position = self.config.get(
            "centering_position", self.sim_surface.INITIAL_CENTERING
        )
        self.clock = pygame.time.Clock()

        self.enabled = True
        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False

    def set_agent_display(self, agent_display: Callable) -> None:
        """
        Set a display callback provided by an agent

        So that they can render their behaviour on a dedicated agent surface, or even on the simulation surface.

        :param agent_display: a callback provided by the agent to display on surfaces
        """
        if EnvViewer.agent_display is None:
            self.extend_display()
        EnvViewer.agent_display = agent_display

    def extend_display(self) -> None:
        if not self.offscreen:
            if self.config["screen_width"] > self.config["screen_height"]:
                self.screen = pygame.display.set_mode(
                    (self.config["screen_width"], 2 * self.config["screen_height"])
                )
            else:
                self.screen = pygame.display.set_mode(
                    (2 * self.config["screen_width"], self.config["screen_height"])
                )
        self.agent_surface = pygame.Surface(
            (self.config["screen_width"], self.config["screen_height"])
        )

    def set_agent_action_sequence(self, actions: list[Action]) -> None:
        """
        Set the sequence of actions chosen by the agent, so that it can be displayed

        :param actions: list of action, following the env's action space specification
        """
        if isinstance(self.env.action_type, DiscreteMetaAction):
            actions = [self.env.action_type.actions[a] for a in actions]
        elif isinstance(self.env.action_type, ContinuousAction):
            actions = [self.env.action_type.get_action(a) for a in actions]
        if len(actions) > 1:
            self.vehicle_trajectory = self.env.vehicle.predict_trajectory(
                actions,
                1 / self.env.config["policy_frequency"],
                1 / 3 / self.env.config["policy_frequency"],
                1 / self.env.config["simulation_frequency"],
            )

    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                EventHandler.handle_event(self.env.action_type, event)

    def display(self) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory, self.sim_surface, offscreen=self.offscreen
            )

        RoadGraphics.display_road_objects(
            self.env.road, self.sim_surface, offscreen=self.offscreen
        )

        if EnvViewer.agent_display:
            EnvViewer.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(
                        self.agent_surface, (0, self.config["screen_height"])
                    )
                else:
                    self.screen.blit(
                        self.agent_surface, (self.config["screen_width"], 0)
                    )

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen,
        )

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(
                self.sim_surface,
                str(self.directory / f"highway-env_{self.frame}.png"),
            )
            self.frame += 1

    def get_image(self) -> np.ndarray:
        """
        The rendered image as a rgb array.

        Gymnasium's channel convention is H x W x C
        """
        surface = (
            self.screen
            if self.config["render_agent"] and not self.offscreen
            else self.sim_surface
        )
        data = pygame.surfarray.array3d(surface)  # in W x H x C channel convention
        return np.moveaxis(data, 0, 1)

    def window_position(self) -> np.ndarray:
        """the world position of the center of the displayed window."""
        if self.observer_vehicle:
            return self.observer_vehicle.position
        elif self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self) -> None:
        """Close the pygame window."""
        pygame.quit()


class EventHandler:
    @classmethod
    def handle_event(
        cls, action_type: ActionType, event: pygame.event.EventType
    ) -> None:
        """
        Map the pygame keyboard events to control decisions

        :param action_type: the ActionType that defines how the vehicle is controlled
        :param event: the pygame event
        """
        if isinstance(action_type, DiscreteMetaAction):
            cls.handle_discrete_action_event(action_type, event)
        elif action_type.__class__ == ContinuousAction:
            cls.handle_continuous_action_event(action_type, event)

    @classmethod
    def handle_discrete_action_event(
        cls, action_type: DiscreteMetaAction, event: pygame.event.EventType
    ) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["FASTER"])
            if event.key == pygame.K_LEFT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["SLOWER"])
            if event.key == pygame.K_DOWN and action_type.lateral:
                action_type.act(action_type.actions_indexes["LANE_RIGHT"])
            if event.key == pygame.K_UP:
                action_type.act(action_type.actions_indexes["LANE_LEFT"])

    @classmethod
    def handle_continuous_action_event(
        cls, action_type: ContinuousAction, event: pygame.event.EventType
    ) -> None:
        action = action_type.last_action.copy()
        steering_index = action_type.space().shape[0] - 1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0.7
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = -0.7
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = -0.7
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0.7
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = 0
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0
        action_type.act(action)


class ObservationGraphics:
    COLOR = (0, 0, 0)

    @classmethod
    def display(cls, obs, sim_surface):
        from highway_env.envs.common.observation import LidarObservation

        if isinstance(obs, LidarObservation):
            cls.display_grid(obs, sim_surface)

        # Display DangerObservation metrics (if available) as overlay text.
        # Local import to avoid circular import issues.
        try:
            from highway_env.envs.common.observation import DangerObservation

            if isinstance(obs, DangerObservation):
                # Obtain normalized danger vector: [ttc_norm, gap_norm, rel_v_norm, a_req_norm, danger_flag, min_ttc_norm]
                danger_vec = obs.observe()
                # Reconstruct (approx.) real-world values when possible
                try:
                    max_ttc = float(obs.max_ttc)
                except Exception:
                    max_ttc = 1.0
                try:
                    max_gap = float(obs.max_gap)
                except Exception:
                    max_gap = 1.0
                try:
                    max_a = float(obs.max_a_req)
                except Exception:
                    max_a = 1.0

                ttc = danger_vec[0] * max_ttc if np.isfinite(max_ttc) else danger_vec[0]
                gap = danger_vec[1] * max_gap
                # rel_v was normalized in [0,1] mapping from [-max_rel_speed, max_rel_speed]
                if hasattr(obs, "max_rel_speed"):
                    rel_v = (danger_vec[2] * 2.0 - 1.0) * obs.max_rel_speed
                else:
                    rel_v = danger_vec[2]
                a_req = danger_vec[3] * max_a
                danger_flag = bool(danger_vec[4] > 0.5)
                min_ttc = danger_vec[5] * max_ttc if np.isfinite(max_ttc) else danger_vec[5]

                # Prepare text lines
                lines = [
                    f"TTC front: {ttc:.2f}s",
                    f"Min TTC: {min_ttc:.2f}s",
                    f"Gap: {gap:.1f}m",
                    f"Rel v: {rel_v:.2f} m/s",
                    f"a_req: {a_req:.2f} m/s²",
                    f"DANGER: {'YES' if danger_flag else 'NO'}",
                ]

                # Draw text on sim_surface (top-left corner)
                # Ensure font is initialized
                try:
                    font = pygame.font.SysFont(None, 18)
                except Exception:
                    pygame.font.init()
                    font = pygame.font.SysFont(None, 18)

                x0, y0 = 10, 10
                color = (255, 80, 80) if danger_flag else (80, 255, 120)
                # Draw a semi-transparent background
                text_height = font.get_linesize() * len(lines) + 6
                bg_rect = pygame.Rect(x0 - 6, y0 - 4, 200, text_height)
                s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                s.fill((0, 0, 0, 100))
                sim_surface.blit(s, (bg_rect.x, bg_rect.y))

                for i, line in enumerate(lines):
                    surf = font.render(line, True, color)
                    sim_surface.blit(surf, (x0, y0 + i * font.get_linesize()))
        except Exception:
            # Don't block rendering if any error occurs while trying to display danger info.
            pass

    @classmethod
    def display_grid(cls, lidar_observation, surface):
        psi = np.repeat(
            np.arange(
                -lidar_observation.angle / 2,
                2 * np.pi - lidar_observation.angle / 2,
                2 * np.pi / lidar_observation.grid.shape[0],
            ),
            2,
        )
        psi = np.hstack((psi[1:], [psi[0]]))
        r = np.repeat(
            np.minimum(lidar_observation.grid[:, 0], lidar_observation.maximum_range), 2
        )
        points = [
            (
                surface.pos2pix(
                    lidar_observation.origin[0] + r[i] * np.cos(psi[i]),
                    lidar_observation.origin[1] + r[i] * np.sin(psi[i]),
                )
            )
            for i in range(np.size(psi))
        ]
        pygame.draw.lines(surface, ObservationGraphics.COLOR, True, points, 1)