from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from random import choices
from typing import Sequence, Union, Tuple


from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from rlgym_sim.utils.state_setters.random_state import RandomState

class BotSetter(StateSetter):
    def __init__(self):
        WeightedSampleSetter(
               (
                  DefaultState(),
                  ReplaySetter("rlgym-ppo/rlgym-ppo-main/rlgym_ppo/ssl_1v1.npy"), #"ssl_1v1.npy", "ssl_2v2.npy", "ssl_3v3.npy"
				  GoaliePracticeState(),
		          WallPracticeState(),
                  AugmentSetter(RandomState(cars_on_ground=True)),
                  AugmentSetter(RandomState(cars_on_ground=False)),       
                    ),
                    (0.10,0.40, 0.20, 0.20, 0.10, 0.10)
                )

    def reset(self, state_wrapper: StateWrapper):
        pass







from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.math import rand_vec3
import numpy as np
from numpy import random as rand
import random

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100

YAW_MAX = np.pi


class GoaliePracticeState(StateSetter):

    def __init__(self, aerial_only=False, allow_enemy_interference=False, first_defender_in_goal=False,
                 reset_to_max_boost=True):
        """
        GoaliePracticeState constructor.

        :param aerial_only: Boolean indicating whether the shots will only be in the air.
        :param allow_enemy_interference: Boolean indicating whether opponents will spawn close enough to easily affect the play
        :param first_defender_in_goal: Boolean indicating whether the first defender will spawn in the goal
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        super().__init__()
        self.team_turn = 0  # swap every reset who's getting shot at

        self.aerial_only = aerial_only
        self.allow_enemy_interference = allow_enemy_interference
        self.first_defender_in_goal = first_defender_in_goal
        self.reset_to_max_boost = reset_to_max_boost

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to set a new shot

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball(state_wrapper, self.team_turn, self.aerial_only)
        self._reset_cars(state_wrapper, self.team_turn, self.first_defender_in_goal, self.allow_enemy_interference,
                         self.reset_to_max_boost)

        # which team will recieve the next incoming shot
        self.team_turn = (self.team_turn + 1) % 2

    def _reset_cars(self, state_wrapper: StateWrapper, team_turn, first_defender_in_goal, allow_enemy_interference,
                    reset_to_max_boost):
        """
        Function to set cars in preparation for an incoming shot

        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's getting shot at
        :param allow_enemy_interference: Boolean indicating whether opponents will spawn close enough to easily affect the play
        :param first_defender_in_goal: Boolean indicating whether the first defender will spawn in the goal
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        first_set = False
        for car in state_wrapper.cars:
            # set random position and rotation for all cars based on pre-determined ranges

            if car.team_num == team_turn and not first_set:
                if first_defender_in_goal:
                    y_pos = -GOAL_LINE if car.team_num == 0 else GOAL_LINE
                    car.set_pos(0, y_pos, z=17)
                    first_set = True
                else:
                    self._place_car_in_box_area(car, team_turn)

            else:
                if allow_enemy_interference:
                    self._place_car_in_box_area(car, team_turn)

                else:
                    self._place_car_in_box_area(car, car.team_num)

            if reset_to_max_boost:
                car.boost = 100

            car.set_rot(0, rand.random() * YAW_MAX - YAW_MAX / 2, 0)

    def _place_car_in_box_area(self, car, team_delin):
        """
        Function to place a car in an allowed areaI 

        :param car: car to be modified
        :param team_delin: team number delinator to look at when deciding where to place the car
        """

        y_pos = (PLACEMENT_BOX_Y - (rand.random() * PLACEMENT_BOX_Y))

        if team_delin == 0:
            y_pos -= PLACEMENT_BOX_Y_OFFSET
        else:
            y_pos += PLACEMENT_BOX_Y_OFFSET

        car.set_pos(rand.random() * PLACEMENT_BOX_X - PLACEMENT_BOX_X / 2, y_pos, z=17)

    def _reset_ball(self, state_wrapper: StateWrapper, team_turn, aerial_only):
        """
        Function to set a new ball towards a goal

        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's getting shot at
        :param aerial_only: Boolean indicating whether should shots only be from the air
        """

        pos, lin_vel, ang_vel = self._get_shot_parameters(team_turn, aerial_only)
        state_wrapper.ball.set_pos(pos[0], pos[1], pos[2])
        state_wrapper.ball.set_lin_vel(lin_vel[0], lin_vel[1], lin_vel[2])
        state_wrapper.ball.set_ang_vel(ang_vel[0], ang_vel[1], ang_vel[2])

    def _get_shot_parameters(self, team_turn, aerial_only):
        """
        Function to set a new ball towards a goal
        
        :param team_turn: team who's getting shot at
        :param aerial_only: Boolean indicating whether should shots only be from the air
        """

        # *** Magic numbers are from manually calibrated shots ***
        # *** They are unrelated to numbers in other functions ***

        shotpick = random.randrange(4)
        INVERT_IF_BLUE = (-1 if team_turn == 0 else 1)  # invert shot for blue

        # random pick x value of target in goal
        x_pos = random.uniform(GOAL_X_MIN, GOAL_X_MAX)

        # if its not an air shot, we can randomize the shot speed
        shot_randomizer = 1 if aerial_only else (random.uniform(.6, 1))

        y_vel = (3000 * INVERT_IF_BLUE) if aerial_only else (3000 * shot_randomizer * INVERT_IF_BLUE)
        if shotpick == 0:  # long range shot

            z_pos = 1500 if aerial_only else random.uniform(100, 1500)

            pos = np.array([x_pos, -3300 * INVERT_IF_BLUE, z_pos])
            lin_vel = np.array([0, y_vel, 600])
        elif shotpick == 1:  # medium range shot
            z_pos = 1550 if aerial_only else random.uniform(100, 1550)

            pos = np.array([x_pos, -500 * INVERT_IF_BLUE, z_pos])
            lin_vel = np.array([0, y_vel, 100])

        elif shotpick == 2:  # angled shot
            z_pos = 1500 if aerial_only else random.uniform(100, 1500)
            x_pos += 3200  # add offset to start the shot from the side
            y_pos = -2000 * INVERT_IF_BLUE

            x_vel = -1100
            y_vel = 2500 * INVERT_IF_BLUE

            pos = np.array([x_pos, y_pos, z_pos])
            lin_vel = np.array([x_vel, y_vel, 650])

        elif shotpick == 3:  # opposite angled shot
            z_pos = 1500 if aerial_only else random.uniform(100, 1500)
            x_pos -= 3200  # add offset to start the shot from the other side
            y_pos = 2000 * INVERT_IF_BLUE

            x_vel = 1100
            y_vel = -2500 * INVERT_IF_BLUE

            pos = np.array([x_pos, y_pos, z_pos])
            lin_vel = np.array([x_vel, y_vel, 650])
        else:
            print("FAULT")

        ang_vel = np.array([0, 0, 0])

        return pos, lin_vel, ang_vel






from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.math import rand_vec3
import numpy as np
from numpy import random as rand
import random

BALL_RADIUS = 94
DEG_TO_RAD = 3.14159265 / 180

class WallPracticeState(StateSetter):

    def __init__(self, air_dribble_odds=1/3, backboard_roll_odds=1/3, side_high_odds=1/3):
        """
        WallPracticeState to setup wall practice
        """
        super().__init__()

        self.air_dribble_odds = air_dribble_odds
        self.backboard_roll_odds = backboard_roll_odds
        self.side_high_odds = side_high_odds

    def reset(self, state_wrapper: StateWrapper):
        choice_list = [0] * int(self.backboard_roll_odds * 100) + \
                      [1] * int(self.side_high_odds * 100) + \
                      [2] * int(self.air_dribble_odds * 100)
        scenario_pick = random.choice(choice_list)

        if scenario_pick == 0:
            self._short_goal_roll(state_wrapper)
        elif scenario_pick == 1:
            self._side_high_roll(state_wrapper)
        elif scenario_pick == 2:
            self._air_dribble_setup(state_wrapper)

    def _air_dribble_setup(self, state_wrapper):
        """
        A medium roll up a side wall with the car facing the roll path

        :param state_wrapper:
        """

        axis_inverter = 1 if random.randrange(2) == 1 else -1
        team_side = 0 if random.randrange(2) == 1 else 1
        team_inverter = 1 if team_side == 0 else -1

        #if only 1 play, team is always 0

        ball_x_pos = 3000 * axis_inverter
        ball_y_pos = random.randrange(7600) - 3800
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + (random.randrange(1000) - 500)) * axis_inverter
        ball_y_vel = random.randrange(1000) * team_inverter
        ball_z_vel = 0
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        chosen_car = [car for car in state_wrapper.cars if car.team_num == team_side][0]
        #if randomly pick, chosen_car is from orange instead

        car_x_pos = 2500 * axis_inverter
        car_y_pos = ball_y_pos
        car_z_pos = 27

        yaw = 0 if axis_inverter == 1 else 180
        car_pitch_rot = 0 * DEG_TO_RAD
        car_yaw_rot = (yaw + (random.randrange(40) - 20)) * DEG_TO_RAD
        car_roll_rot = 0 * DEG_TO_RAD

        chosen_car.set_pos(car_x_pos, car_y_pos, car_z_pos)
        chosen_car.set_rot(car_pitch_rot, car_yaw_rot, car_roll_rot)
        chosen_car.boost = 100

        for car in state_wrapper.cars:
            if car is chosen_car:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927 / 180), 0)

    def _side_high_roll(self, state_wrapper):
        """
        A high vertical roll up the side of the field

        :param state_wrapper:
        """
        sidepick = random.randrange(2)

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1


        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        ball_x_pos = 3000 * side_inverter
        ball_y_pos = random.randrange(1500) - 750
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + random.randrange(1000) - 500) * side_inverter
        ball_y_vel = random.randrange(1500) - 750
        ball_z_vel = random.randrange(300)
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car_blue = [car for car in state_wrapper.cars if car.team_num == 0][0]

        #blue car setup
        blue_pitch_rot = 0 * DEG_TO_RAD
        blue_yaw_rot = 90 * DEG_TO_RAD
        blue_roll_rot = 90 * side_inverter * DEG_TO_RAD
        wall_car_blue.set_rot(blue_pitch_rot, blue_yaw_rot, blue_roll_rot)

        blue_x = 4096 * side_inverter
        blue_y = -2500 + (random.randrange(500) - 250)
        blue_z = 600 + (random.randrange(400) - 200)
        wall_car_blue.set_pos(blue_x, blue_y, blue_z)
        wall_car_blue.boost = 100

        #orange car setup
        wall_car_orange = None
        if len(state_wrapper.cars) > 1:
            wall_car_orange = [car for car in state_wrapper.cars if car.team_num == 1][0]
            # orange car setup
            orange_pitch_rot = 0 * DEG_TO_RAD
            orange_yaw_rot = -90 * DEG_TO_RAD
            orange_roll_rot = -90 * side_inverter * DEG_TO_RAD
            wall_car_orange.set_rot(orange_pitch_rot, orange_yaw_rot, orange_roll_rot)

            orange_x = 4096 * side_inverter
            orange_y = 2500 + (random.randrange(500) - 250)
            orange_z = 400 + (random.randrange(400) - 200)
            wall_car_orange.set_pos(orange_x, orange_y, orange_z)
            wall_car_orange.boost = 100

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car_orange or car is wall_car_blue:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927/180), 0)

    def _short_goal_roll(self, state_wrapper):
        """
        A short roll across the backboard and down in front of the goal

        :param state_wrapper:
        :return:
        """

        if len(state_wrapper.cars) > 1:
            defense_team = random.randrange(2)
        else:
            defense_team = 0
        sidepick = random.randrange(2)

        defense_inverter = 1
        if defense_team == 0:
            # change side
            defense_inverter = -1

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        x_random = random.randrange(446)
        ball_x_pos = (-2850 + x_random) * side_inverter
        ball_y_pos = (5120 - BALL_RADIUS) * defense_inverter
        ball_z_pos = 1400 + random.randrange(400) - 200
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (1000 + random.randrange(400) - 200) * side_inverter
        ball_y_vel = 0
        ball_z_vel = 550
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)


        wall_car = [car for car in state_wrapper.cars if car.team_num == defense_team][0]

        wall_car_x = (2000 - random.randrange(500)) * side_inverter
        wall_car_y = 5120 * defense_inverter
        wall_car_z = 1000 + (random.randrange(500) - 500)
        wall_car.set_pos(wall_car_x, wall_car_y, wall_car_z)

        wall_pitch_rot = (0 if side_inverter == -1 else 180) * DEG_TO_RAD
        wall_yaw_rot = 0 * DEG_TO_RAD
        wall_roll_rot = -90 * defense_inverter * DEG_TO_RAD
        wall_car.set_rot(wall_pitch_rot, wall_yaw_rot, wall_roll_rot)
        wall_car.boost = 25

        if len(state_wrapper.cars) > 1:
            challenge_car = [car for car in state_wrapper.cars if car.team_num != defense_team][0]
            challenge_car.set_pos(0, 1000 * defense_inverter, 0)

            challenge_pitch_rot = 0 * DEG_TO_RAD
            challenge_yaw_rot = 90 * defense_inverter * DEG_TO_RAD
            challenge_roll_rot = 0 * DEG_TO_RAD
            challenge_car.set_rot(challenge_pitch_rot, challenge_yaw_rot, challenge_roll_rot)
            challenge_car.boost = 100

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car or car is challenge_car:
                continue

            car.set_pos(random.randrange(2944) - 1472, (-4500 + random.randrange(500) - 250) * defense_inverter, 0)
            car.set_rot(0, (random.randrange(360) - 180) * DEG_TO_RAD, 0)


