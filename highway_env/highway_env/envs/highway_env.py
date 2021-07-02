import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 60,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "COLLISION_REWARD": 100,
            "right_lane_reward": 0.3,
            "HIGH_SPEED_REWARD": 1,
            "HEADWAY_COST": 1,
            "HEADWAY_TIME": 1.2,
            "lane_change_reward": 0,
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reward(self, action: int) -> float:
        return self._agent_reward(action)

    def _agent_reward(self, action: int) -> float:
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        headway_distance = self._compute_headway_distance(self.vehicle)
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * self.vehicle.speed)) if self.vehicle.speed > 0 else 0

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]

        reward = self.config["COLLISION_REWARD"] * (-1 * self.vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
                 + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
        return reward

    def _compute_headway_distance(self, vehicle, ):
        headway_distance = 60
        for v in self.road.vehicles:
            if (v.lane_index == vehicle.lane_index) and (v.position[0] > vehicle.position[0]):
                hd = v.position[0] - vehicle.position[0]
                if hd < headway_distance:
                    headway_distance = hd

        return headway_distance

    def _reset(self, ) -> None:
        if self.episode == 1:
            self.num_HDVs = self.config["vehicles_count"]
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # print(self.num_HDVs)
        other_per_controlled = near_split(self.num_HDVs, num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=27,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                self.road.vehicles.append(
                    other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                )

    def sample_tasks(self, num_tasks):
        num_HDVs = np.random.randint(40, 50, size=(num_tasks, 1))
        tasks = [{'num_HDVs': n[0]} for n in num_HDVs]
        return tasks

    def reset_task(self, task):
        self.num_HDVs = task['num_HDVs']

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)
