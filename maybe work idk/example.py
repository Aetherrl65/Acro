import gym
import numpy as np
import rlgym_sim as rlgym
import os
from numpy import load
from typing import List, Union
from typing import Any
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from typing import Any
from gym.spaces import Discrete
from rlgym_sim.utils import ActionParser
from turtle import pd
from rlgym_sim.utils.obs_builders import AdvancedObs

# start a new wandb run to track this script
#wandb.init(
    # set the wandb project where this run will be logged
#    project="rlgym-ppo")

#path = r"C:\Users\Colton Brown\Desktop\ACRO\rlgym-ppo\rlgym-ppo-main\rlgym_ppo\states_scores_duels.npz"
#assert os.path.isfile(path)
#with open(path, "r") as f:
#    pass

class LookupAction(ActionParser):
    def __init__(self, bins=None):
        super().__init__()
        if bins is None:
            self.bins = [(-1, 0, 1)] * 5
        elif isinstance(bins[0], (float, int)):
            self.bins = [bins] * 5
        else:
            assert len(bins) == 5, "Need bins for throttle, steer, pitch, yaw and roll"
            self.bins = bins
        self._lookup_table = self.make_lookup_table(self.bins)

    @staticmethod
    def make_lookup_table(bins):
        actions = []
        # Ground
        for throttle in bins[0]:
            for steer in bins[1]:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in bins[2]:
            for yaw in bins[3]:
                for roll in bins[4]:
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        indexes = np.array(actions, dtype=np.int32)
        indexes = np.squeeze(indexes)
        return self._lookup_table[indexes]
    

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)



def build_rocketsim_env():
    import rlgym_sim
    import random
    from rlgym_sim.utils.state_setters.random_state import RandomState
    from rlgym_sim.utils.state_setters.weighted_state_setter import BotSetter
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, TouchBallReward, FaceBallReward, \
        EventReward, DefenseTrainer, FlipCorrecter, LandingRecoveryReward, TouchVelChange, AerialTraining, AerialNavigation, VelocityBallToGoalReward, AirReward, PositiveBallVelToGoalReward, PositivePlayerVelToBallReward, PowerShotReward, DefenseTrainer, ModifiedTouchReward, SpeedReward, HeightTouchReward, WallTouchReward, LavaFloorReward, SaveBoostReward, JumpTouchReward, AlignBallGoal, VelocityReward, AerialDistanceReward, LiuDistancePlayerToBallReward, LiuDistanceBallToGoalReward, RewardIfTouchedLast
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import DiscreteAction
    from rlgym_sim.utils.obs_builders import AdvancedObs

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    rewards_to_combine = (VelocityPlayerToBallReward(),
                          VelocityBallToGoalReward(),
                          TouchBallReward(),
                          FaceBallReward(),
                          TouchVelChange(),
                          WallTouchReward(),
                          JumpTouchReward(),
                          SaveBoostReward(),
                          AerialDistanceReward(),
                          PowerShotReward(),
                          AlignBallGoal(),
                          LavaFloorReward(),
                          EventReward(team_goal=15, concede=-7, demo=0.2, boost_pickup=0.1))
    reward_weights = (1, 5, 3, 1, 4, 2, 5, 1, 5, 5, 1, 0.001, 17.0)

    reward_fn = CombinedReward(reward_functions=rewards_to_combine,
                               reward_weights=reward_weights)

    #obs_builder = DefaultObs(
     #   pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
     #   ang_coef=1 / np.pi,
     #   lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
     #   ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    
    obs_builder = AdvancedObs()
    
    random_state = RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=False)
    setter = BotSetter(),

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=random_state
                         )

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # 32 processes
    n_proc = 14

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=100000,
                      ts_per_iteration=100000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1_000_000_000,
                      log_to_wandb=True)

    learner.learn()
