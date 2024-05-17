import numpy as np
from math import sqrt
from rlgym_sim.utils.gamestates.physics_object import PhysicsObject


def sign(x: float) -> int:
    return -1 if x < 0 else 1


def clamp(max_range: float, min_range: float, number: float) -> float:
    return max((min_range, min((number, max_range))))


def normalize(x: np.array) -> np.array:
    norm = np.linalg.norm(x)
    if norm == 0:
       return x
    return x / norm


def distance(x: np.array, y: np.array) -> float:
    return np.linalg.norm(x - y)


def distance2D(x: np.array, y: np.array)->float:
    x[2] = 0
    y[2] = 0
    return distance(x, y)


def relative_velocity(vec1: np.array, vec2: np.array)-> np.array:
    return vec1 - vec2


def relative_velocity_mag(vec1: np.array, vec2: np.array)-> float:
    return np.linalg.norm(relative_velocity(vec1, vec2))


def simple_physics_object_mirror(base: PhysicsObject) -> dict:
    """ mirrors data on the y axis """
    return {
        'position': np.array([-base.postion[0], base.position[1], base.postion[2]]),
        'linear_velocity': np.array([-base.linear_velocity[0], base.position[1], base.linear_velocity[2]]),
        'angular_velocity': np.array([-base.angular_velocity[0], base.angular_velocity[1], -base.angular_velocity[2]])
    }


if __name__ == "__main__":
    pass






import numpy as np
from rlgym_sim.utils import RewardFunction, math
from rlgym_sim.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils import math as rl_math

SIDE_WALL_X = 4096  # +/-
BACK_WALL_Y = 5120  # +/-
CEILING_Z = 2044
BACK_NET_Y = 6000  # +/-

GOAL_HEIGHT = 642.775

ORANGE_GOAL_CENTER = (0, BACK_WALL_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_CENTER = (0, -BACK_WALL_Y, GOAL_HEIGHT / 2)

# Often more useful than center
ORANGE_GOAL_BACK = (0, BACK_NET_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_BACK = (0, -BACK_NET_Y, GOAL_HEIGHT / 2)

# ORANGE_GOAL_WAY_BACK = (0, 8000, GOAL_HEIGHT / 2)
# BLUE_GOAL_WAY_BACK = (0, -8000, GOAL_HEIGHT / 2)
# [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
BALL_RADIUS = 92.75

BALL_MAX_SPEED = 6000
CAR_MAX_SPEED = 2300
SUPERSONIC_THRESHOLD = 2200
CAR_MAX_ANG_VEL = 5.5

BLUE_TEAM = 0
ORANGE_TEAM = 1
NUM_ACTIONS = 8

BOOST_LOCATIONS = (
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (-940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
)

class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, norm_vel))


class FlipCorrecter(RewardFunction):
    def __init__(self) -> None:
        self.last_velocity = np.zeros(3)
        self.armed = False

    def reset(self, initial_state: GameState) -> None:
        self.last_velocity = np.zeros(3)
        self.armed = False

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if self.armed and player.on_ground:
            self.armed = False

        elif self.armed and not player.has_flip:
            self.armed = False
            if distance(player.car_data.position, state.ball.position) <= 500:
                vel_diff = player.car_data.linear_velocity - self.last_velocity
                if np.linalg.norm(vel_diff) > 100 and previous_action[5] == 1:
                    ball_dir = normalize(state.ball.position - player.car_data.position)
                    reward = ball_dir.dot(normalize(vel_diff))
                # if distance(player.car_data.position, state.ball.position) >= 1200:
                #     rew2 = normalize(self.last_velocity).dot(normalize(vel_diff))
                #     if rew2 > reward:
                #         reward = rew2

        elif not self.armed and not player.on_ground and player.has_flip:
            self.armed = True

        self.last_velocity = player.car_data.linear_velocity
        return reward

class LandingRecoveryReward(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.up = np.array([0, 0, 1])

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and player.car_data.linear_velocity[2] < 0
            and player.car_data.position[2] > 250
        ):
            flattened_vel = normalize(
                np.array(
                    [
                        player.car_data.linear_velocity[0],
                        player.car_data.linear_velocity[1],
                        0,
                    ]
                )
            )
            forward = player.car_data.forward()
            flattened_forward = normalize(np.array([forward[0], forward[1], 0]))
            reward += flattened_vel.dot(flattened_forward)
            reward += self.up.dot(player.car_data.up())
            reward /= 2

        return reward


class LiuDistancePlayerToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        return np.exp(-0.5 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        car_speed = np.linalg.norm(player.car_data.linear_velocity)
        car_dir = sign(player.car_data.forward().dot(player.car_data.linear_velocity))
        if car_dir < 0:
            car_speed /= -2300

        else:
            car_speed /= 2300
        return min(car_speed, 1)
    
class AerialTraining(RewardFunction):
    def __init__(self, ball_height_min=400, player_min_height=300) -> None:
        super().__init__()
        self.vel_reward = VelocityPlayerToBallReward()
        self.ball_height_min = ball_height_min
        self.player_min_height = player_min_height

    def reset(self, initial_state: GameState) -> None:
        self.vel_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
                not player.on_ground
                and state.ball.position[2] > self.ball_height_min
                and self.player_min_height < player.car_data.position[2] < state.ball.position[2]
        ):
            divisor = max(1, distance(player.car_data.position, state.ball.position)/1000)
            return self.vel_reward.get_reward(player, state, previous_action)/divisor

        return 0

class AerialNavigation(RewardFunction):
    def __init__(
        self, ball_height_min=400, player_height_min=200, beginner=True
    ) -> None:
        super().__init__()
        self.ball_height_min = ball_height_min
        self.player_height_min = player_height_min
        self.face_reward = FaceBallReward()
        self.beginner = beginner

    def reset(self, initial_state: GameState) -> None:
        self.face_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and state.ball.position[2]
            > self.ball_height_min
            > player.car_data.position[2]
            and player.car_data.linear_velocity[2] > 0
            and distance2D(player.car_data.position, state.ball.position)
            < state.ball.position[2] * 3
        ):
            # vel check
            ball_direction = normalize(state.ball.position - player.car_data.position)
            alignment = ball_direction.dot(normalize(player.car_data.linear_velocity))
            if self.beginner:
                reward += max(
                    0, alignment * 0.5
                )  # * (np.linalg.norm(player.car_data.linear_velocity)/2300)
                # old
                # #face check
                # face_reward = self.face_reward.get_reward(player, state, previous_action)
                # if face_reward > 0:
                #     reward += face_reward * 0.25
                # #boost check
                #     if previous_action[6] == 1 and player.boost_amount > 0:
                #         reward += face_reward

            reward += alignment * (
                np.linalg.norm(player.car_data.linear_velocity) / 2300.0
            )

        return max(reward, 0)


class TouchVelChange(RewardFunction):
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = vel_difference / 4600.0

        self.last_vel = state.ball.linear_velocity

        return reward

import math

class WallTouchReward(RewardFunction):
    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp
        self.max = math.inf

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and player.on_ground and state.ball.position[2] >= self.min_height:
            return (clamp(self.max, 0.0001, state.ball.position[2] - 92) ** self.exp)-1

        return 0

class HeightTouchReward(RewardFunction):
    def __init__(self, min_height=92, exp=0.2, coop_dist=0):
        super().__init__()
        self.min_height = min_height
        self.exp = exp
        self.cooperation_dist = coop_dist

    def reset(self, initial_state: GameState):
        pass

    def cooperation_detector(self, player: PlayerData, state: GameState):
        for p in state.players:
            if p.car_id != player.car_id and \
                    distance(player.car_data.position, p.car_data.position) < self.cooperation_dist:
                return True

        return False


    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            if state.ball.position[2] >= self.min_height:
                if not player.on_ground or self.cooperation_dist < 90 or not self.cooperation_detector(player, state):
                    if player.on_ground:
                        reward += clamp(5000, 0.0001, (state.ball.position[2]-92)) ** self.exp
                    else:
                        reward += clamp(500, 1, (state.ball.position[2] ** (self.exp*2)))

            elif not player.on_ground:
                reward += 1

        return reward


class LavaFloorReward(RewardFunction):
    @staticmethod
    def get_reward(player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        if player.on_ground and player.car_data.position[2] < 50:
            return -1
        return 0

    @staticmethod
    def reset(initial_state: GameState):
        pass

class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return (inv_t)
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            return float(np.dot(norm_pos_diff, norm_vel))

class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))


class ModifiedTouchReward(RewardFunction):
    def __init__(self, min_change: float = 300, min_height: float = 200, vel_scale: float = 1, touch_scale: float = 1, jump_reward: bool = False, jump_scale: float = 0.1, tick_min: int = 0):
        super().__init__()
        self.psr = PowerShotReward(min_change)
        self.min_height = min_height
        self.height_cap = 2044-92.75
        self.vel_scale = vel_scale
        self.touch_scale = touch_scale
        self.jump_reward = jump_reward
        self.jump_scale = jump_scale
        self.tick_count = 0
        self.tick_min = tick_min

    def reset(self, initial_state: GameState):
        self.psr.reset(initial_state)
        self.tick_count = 0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        psr = self.psr.get_reward(player, state, previous_action)
        if player.ball_touched:
            if self.tick_count <= 0:
                self.tick_count = self.tick_min
                reward += abs(psr * self.vel_scale)

                if not player.on_ground:
                    if self.jump_reward:
                        reward += self.jump_scale
                        if not player.has_flip:
                            reward += self.jump_scale
                    if state.ball.position[2] > self.min_height:
                        reward += abs((state.ball.position[2]/self.height_cap) * self.touch_scale)
            else:
                self.tick_count -= 1
        else:
            self.tick_count -= 1

        return reward

class PositiveBallVelToGoalReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.rew = VelocityBallToGoalReward()

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return clamp(1, 0, self.rew.get_reward(player, state, previous_action))

class PositivePlayerVelToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.rew = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return clamp(1, 0, self.rew.get_reward(player, state, previous_action))

class PowerShotReward(RewardFunction):
    def __init__(self, min_change: float = 300):
        super().__init__()
        self.min_change = min_change
        self.last_velocity = np.array([0, 0])

    def reset(self, initial_state: GameState):
        self.last_velocity = np.array([0, 0])

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        cur_vel = np.array(
            [state.ball.linear_velocity[0], state.ball.linear_velocity[1]]
        )
        if player.ball_touched:
            vel_change = rl_math.vecmag(self.last_velocity - cur_vel)
            if vel_change > self.min_change:
                reward = vel_change / (2300*2)

        self.last_velocity = cur_vel
        return reward

class DefenseTrainer(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            defense_objective = np.array(BLUE_GOAL_BACK)
        else:
            defense_objective = np.array(ORANGE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = defense_objective - state.ball.position
        norm_pos_diff = normalize(pos_diff)
        vel = vel/BALL_MAX_SPEED
        scale = clamp(1, 0, 1 - (distance2D(state.ball.position, defense_objective)/10000))
        return -clamp(1, 0, float(norm_pos_diff.dot(vel)*scale))

#import math

#class DribbleReward(RewardFunction):
#    def reset(self, initial_state: GameState):
#        pass
#
#    def get_reward(self, player: PlayerData, state: GameState, prevAction: np.ndarray) -> float:
#        MIN_BALL_HEIGHT = 109.0
#        MAX_BALL_HEIGHT = 180.0
#        MAX_DISTANCE = 197.0
#        SPEED_MATCH_FACTOR = 2.0  # Adjust this value to control the importance of speed matching
#        CAR_MAX_SPEED = CommonValues.CAR_MAX_SPEED
#        if player.carState.isOnGround and MIN_BALL_HEIGHT <= state.ball.pos.z <= MAX_BALL_HEIGHT and math.sqrt((player.phys.pos - state.ball.pos).Length()) < MAX_DISTANCE:
#            player_speed = player.phys.vel.Length()
#            ball_speed = state.ball.vel.Length()
#            speed_match_reward = ((player_speed / CAR_MAX_SPEED) + SPEED_MATCH_FACTOR * (1.0 - abs(player_speed - ball_speed) / (player_speed + ball_speed))) / 2.0
#            return speed_match_reward  # Reward for successful dribbling, with a bonus for speed matching, normalized to 1
#        else:
#            return 0.0  # No reward

class AirReward(RewardFunction):
    def __init__(
        self,
    ):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if not player.on_ground:
            if player.has_flip:
                return 0.5
            else:
                return 1
        return 0


if __name__ == "__main__":
    pass


class TouchBallReward(RewardFunction):
    def __init__(self, aerial_weight=0.):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            # Default just rewards 1, set aerial weight to reward more depending on ball height
            return ((state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)) ** self.aerial_weight
        return 0
    
class TouchBallReward2(RewardFunction):
    def __init__(self, aerial_weight=5., velocity_weight=10.):
        self.aerial_weight = aerial_weight
        self.velocity_weight = velocity_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            # Calculate the velocity of the ball
            ball_velocity = np.linalg.norm(state.ball.linear_velocity)

            # Reward based on aerial height and velocity
            reward = ((state.ball.position + BALL_RADIUS) / (2 * BALL_RADIUS)) ** self.aerial_weight
            reward += ball_velocity ** self.velocity_weight

            return reward
        return 0

class JumpTouchReward(RewardFunction):
    """
    a ball touch reward that only triggers when the agent's wheels aren't in contact with the floor
    adjust minimum ball height required for reward with 'min_height' as well as reward scaling with 'exp'
    """

    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
            ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return ((state.ball.position[2] - 92) ** self.exp)-1

        return 0
    
class DiffReward(RewardFunction):
    
    def __init__(self, reward_function: RewardFunction, negative_slope=1.):
        super().__init__()
        self.reward_function = reward_function
        self.last_values = {}
        self.negative_slope = negative_slope  # Can weight negative values differently

    def reset(self, initial_state: GameState):
        self.last_values = {}

    def _calculate_diff(self, player, rew):
        last = self.last_values.get(player.car_id)
        self.last_values[player.car_id] = rew
        if last is not None:
            ret = rew - last
            return self.negative_slope * ret if ret < 0 else ret
        else:
            return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_reward(player, state, previous_action)
        return self._calculate_diff(player, rew)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = self.reward_function.get_final_reward(player, state, previous_action)
        return self._calculate_diff(player, rew)
    

import numpy as np

from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.common_values import BACK_WALL_Y
from rlgym_sim.utils.gamestates import PlayerData, GameState
from typing import Optional

RAMP_HEIGHT = 256
UP_VECTOR = np.array([0.0, 0.0, 1.0])
G_CONST = np.array([0.0, 0.0, -650.0])

class AerialDistanceReward(RewardFunction):
    def __init__(self, height_scale: float = 10, distance_scale: float = 10, scale_by_upness: bool = False, tick_skip: int = 0):
        super().__init__()
        self.height_scale = height_scale
        self.distance_scale = distance_scale

        self.current_car: Optional[PlayerData] = None
        self.prev_state: Optional[GameState] = None
        self.ball_distance: float = 0
        self.car_distance: float = 0
        self.scale_by_upness = scale_by_upness
        self.tick_skip = tick_skip

        if scale_by_upness and tick_skip == 0:
            raise ValueError("Must assign tick_skip when using scale_by_upness")

    def reset(self, initial_state: GameState):
        self.current_car = None
        self.prev_state = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0
        is_current = self.current_car is not None and self.current_car.car_id == player.car_id
        # Test if player is on the ground
        if player.car_data.position[2] < RAMP_HEIGHT:
            if is_current:
                is_current = False
                self.current_car = None
        # First non ground touch detection
        elif player.ball_touched and not is_current:
            is_current = True
            self.ball_distance = 0
            self.car_distance = 0
            rew = self.height_scale * max(player.car_data.position[2] + state.ball.position[2] - 2 * RAMP_HEIGHT, 0)

        # Still off the ground after a touch, add distance and reward for more touches
        elif is_current:
            self.car_distance += np.linalg.norm(player.car_data.position - self.current_car.car_data.position)
            self.ball_distance += np.linalg.norm(state.ball.position - self.prev_state.ball.position)
            # Cash out on touches
            if player.ball_touched:
                upness = 1

                if self.scale_by_upness:
                    delta_v_due_to_gravity = G_CONST * self.tick_skip / 120.0
                    delta_v = self.prev_state.ball.linear_velocity - state.ball.linear_velocity - delta_v_due_to_gravity
                    accel_vector = delta_v / np.linalg.norm(delta_v)
                    upness = np.dot(accel_vector, UP_VECTOR)

                    # make "upness" be bounded by [0, 1]
                    upness = np.clip(upness, 0, 0.8) * 1.25

                rew = upness * self.distance_scale * (self.car_distance + self.ball_distance)
                self.car_distance = 0
                self.ball_distance = 0

        if is_current:
            self.current_car = player  # Update to get latest physics info

        self.prev_state = state

        return rew / (2 * BACK_WALL_Y)