from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import random
import time
from abc import ABC
from os import path

import csv
import pygame
import tensorflow as tf
import numpy as np
import pygame.gfxdraw
from tensorflow import keras
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from gym_game.envs.game_controls import GameControls
from gym_game.envs.models.game_object import Target, Attacker, Defender
from gym_game.envs.models.physics.physics import get_pos_in_pixel

tf.compat.v1.enable_v2_behavior()


class CatchMeIfYouCanEnv(py_environment.PyEnvironment, ABC):

    def __init__(self):
        # /***************************************************************************
        # Environment
        # /***************************************************************************
        super().__init__()
        self.last_action = 0
        self.same_buffer = 0
        self._state = None
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=9, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(11,), dtype=np.float, minimum=[0, 0, 0, 0, 0, 0, -2000, -2000, 0, 1, 0],
            maximum=[100, 100, 100, 100, 100, 100, 2000, 2000, 1, 8, 100], name='observation')
        self._episode_ended = False
        self.javris_prediction: keras.models.Sequential = None

        # /***************************************************************************
        # Game Variables
        # /***************************************************************************
        self.episode_length = 2400
        self.game_start = time.time()
        # Latencies in ms
        self.latencies = [10, 50, 100, 150, 200, 250, 300]
        self.time_delay_last_time = time.time()
        self.game_delay = 0.05  # this ensures that the game runs at 20Hz for the real robot controll
        self._human_mode = False
        self._game_controls: GameControls = None
        self._real_academy = False
        self._pygame = pygame
        self.whole_reward = 0
        self.game_reward = 0
        self.level = 1
        self.last_hits = 0
        self.hits = 0
        # If Robot is in firing = 1; else = 0
        self.firing = 0
        self.firing_buffer = 0
        self.iterations = 0
        self.last_state = None
        self.last_time = round(time.time() * 1000)
        self.last_hit_time = time.time()
        self.last_measurement_time = time.time()
        self.screen = None
        self.movement_step = 2
        self.background_image_path = path.join(path.dirname(__file__), "assets/2f0.png")
        self.defender_size = 60
        self.attacker_size = 30
        self.window = {"x": 640, "y": 480}
        self.game_boundaries = {"x_min": 0 + self.defender_size, "y_min": 0 + self.defender_size,
                                "x_max": self.window.get("x") - self.defender_size,
                                "y_max": self.window.get("y") - self.defender_size}
        self.game_boundaries_percent = {"x_min": 10, "y_min": 10, "x_max": 80, "y_max": 80}
        # /***************************************************************************
        #  * Target definition
        #  ***************************************************************************/
        target_image_path = path.join(path.dirname(__file__), "assets/attacker_gray.png")
        # Start somewhere between 10% and 80% within the screen
        self.t_x = random.randint(self.game_boundaries_percent.get("x_min"), self.game_boundaries_percent.get("x_max"))
        self.t_y = random.randint(self.game_boundaries_percent.get("y_min"), self.game_boundaries_percent.get("y_max"))
        self.t_v_max = 1
        self.target: Target = Target(
            height=self.defender_size,
            image=target_image_path,
            x=self.t_x,
            y=self.t_y,
            x_min=self.game_boundaries_percent.get("x_min"),
            y_min=self.game_boundaries_percent.get("y_min"),
            x_max=self.game_boundaries_percent.get("x_max"),
            y_max=self.game_boundaries_percent.get("y_max"),
            v_max=self.t_v_max,
            step=self.movement_step
        )

        # /***************************************************************************
        #  * Attacker definition
        #  ***************************************************************************/
        attacker_image = path.join(path.dirname(__file__), "assets/Attacker.png")
        self.a_x = random.randint(self.game_boundaries_percent.get("x_min"), self.game_boundaries_percent.get("x_max"))
        self.a_y = random.randint(self.game_boundaries_percent.get("y_min"), self.game_boundaries_percent.get("y_max"))
        self.a_x_dot = 0
        self.a_y_dot = 0
        self.a_x_dot_dot = 0
        self.a_y_dot_dot = 0
        self.a_v_max = 1
        self.a_a_max = 1
        self.attacker: Attacker = Attacker(
            target=self.target,
            image=attacker_image,
            x=self.a_x,
            y=self.a_y,
            height=70,
            v_max=self.a_v_max,
            a_max=self.a_a_max,
            x_min=self.game_boundaries_percent.get("x_min"),
            y_min=self.game_boundaries_percent.get("y_min"),
            x_max=self.game_boundaries_percent.get("x_max"),
            y_max=self.game_boundaries_percent.get("y_max"),
            game_controls=self.game_controls,
            step=1.3)

        # Defender Definition
        defender_image = path.join(path.dirname(__file__), "assets/defender.png")
        self.d_x = random.randint(self.game_boundaries_percent.get("x_min"), self.game_boundaries_percent.get("x_max"))
        self.d_y = random.randint(self.game_boundaries_percent.get("y_min"), self.game_boundaries_percent.get("y_max"))
        self.d_x_dot = 0
        self.d_y_dot = 0
        self.d_v_max = 0.5
        self.d_a_max = 0.5
        self.defender: Defender = Defender(
            image=defender_image,
            height=self.defender_size,
            v_max=self.d_v_max,
            a_max=self.d_a_max,
            x=self.d_x,
            y=self.d_y,
            x_dot=self.d_x_dot,
            y_dot=self.d_y_dot,
            x_min=self.game_boundaries_percent.get("x_min"),
            y_min=self.game_boundaries_percent.get("y_min"),
            x_max=self.game_boundaries_percent.get("x_max"),
            y_max=self.game_boundaries_percent.get("y_max"),
            game_controls=self.game_controls
        )
        self._state = self._get_obs()

    @property
    def real_academy(self):
        return self._real_academy

    @property
    def pygame(self):
        return self._pygame

    @property
    def human_mode(self):
        return self._human_mode

    @human_mode.setter
    def human_mode(self, game_var):
        self._human_mode = game_var

    @property
    def game_controls(self):
        return self._game_controls

    @pygame.setter
    def pygame(self, game_var):
        self._pygame = game_var

    @real_academy.setter
    def real_academy(self, ra):
        self._real_academy = ra

    @game_controls.setter
    def game_controls(self, gc):
        self._game_controls = gc
        self.attacker.game_controls = self.game_controls
        self.defender.game_controls = self.game_controls

    def set_game_controls(self, gc):
        self._game_controls = gc
        self.attacker.game_controls = self.game_controls
        self.defender.game_controls = self.game_controls

    def set_javris_prediction(self, javris_prediction):
        self.javris_prediction = javris_prediction

    def set_academy(self, ac):
        self._real_academy = ac

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_obs(self):
        current_time = round(time.time() * 1000)
        x_a, y_a = self.attacker.get_obs()
        prediction = []
        x_a_dest, y_a_dest = self.attacker.target.get_obs()
        x_d, y_d = self.defender.get_obs()
        if self._state is not None and current_time > self.last_time:
            x_dot_d = (x_d - self._state[4]) / (current_time - self.last_time)
            y_dot_d = (y_d - self._state[5]) / (current_time - self.last_time)
        else:
            x_dot_d = 0
            y_dot_d = 0
        self.last_time = current_time

        if self.game_controls:
            return np.array([x_a, y_a, x_a_dest, y_a_dest, x_d, y_d, x_dot_d, y_dot_d, self.firing, self.level,
                             self.game_controls.ping])
        else:
            return np.array([x_a, y_a, x_a_dest, y_a_dest, x_d, y_d, x_dot_d, y_dot_d, self.firing, self.level, 0])

    def reposition(self):
        a_x = random.uniform(self.game_boundaries_percent.get("x_min"), self.game_boundaries_percent.get("x_max"))
        a_y = random.uniform(self.game_boundaries_percent.get("y_min"), self.game_boundaries_percent.get("y_max"))
        t_x = random.uniform(self.game_boundaries_percent.get("x_min"), self.game_boundaries_percent.get("x_max"))
        t_y = random.uniform(self.game_boundaries_percent.get("y_min"), self.game_boundaries_percent.get("y_max"))

        # Defender Definition
        d_x = random.uniform(self.game_boundaries_percent.get("x_min"), self.game_boundaries_percent.get("x_max"))
        d_y = random.uniform(self.game_boundaries_percent.get("y_min"), self.game_boundaries_percent.get("y_max"))

        self.attacker.position["x"] = a_x
        self.attacker.position["y"] = a_y
        self.target.position["x"] = t_x
        self.target.position["y"] = t_y
        self.defender.position["x"] = d_x
        self.defender.position["y"] = d_y
        self._state = self._get_obs()

    def _reset(self):
        self.whole_reward = 0
        self.same_buffer = 0
        self.reposition()
        self.level = 1
        if self.game_controls:
            self.game_controls.hits_since_start = 0
            self.game_controls.level = self.level
        self.hits = 0
        self.last_hit_time = time.time()
        self.game_score = 0
        self.game_start = time.time()
        self.iterations = 0
        self.last_hits = 0
        self._episode_ended = False
        return ts.restart([self._state])

    def _is_done(self):
        if self.real_academy and (self.level >= 8 or self.iterations > 2400):
            return True
        else:
            return False

    def save_step(self):
        if not self.javris_prediction:
            file_name = "./steps_user.csv"
            file_exists = os.path.isfile(file_name)
            with open(file_name,'a') as csvfile:
                headers = ['Index','Action','x_a','y_a','x_c','y_c','x_d','y_d','v_x_d','v_y_d','firing','level','ping']
                writer = csv.DictWriter(csvfile, delimiter=",",lineterminator="\n", fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'Index': self.iterations,'Action': self.last_action, 'x_a': self._state[0],'y_a': self._state[1],'x_c':self._state[2],'y_c':self._state[3],'x_d':self._state[4],'y_d':self._state[5],'v_x_d':self._state[6],'v_y_d':self._state[7],'firing':self.firing,'level': self.level, 'ping': self.game_controls.ping})

    def _step(self, action):
        self.iterations += 1
        self.is_same_state()
        if self._episode_ended:
            time.sleep(3)
            return self.reset()

        # Make sure episodes don't go on forever.
        if self._is_done():
            print("EPISODE_ENDED")
            self._episode_ended = True
        else:
            self.last_state = self._state
            # x_a, y_a, x_a_dest, y_a_dest, x_d, y_d, x_dot_d, y_dot_d  = state_array
            self.target.move(action=action)
            if self.firing == 1:
                if self.firing_buffer < 30:
                    self.firing_buffer += 1
                else:
                    self.firing = 0
                    self.firing_buffer = 0
                    if not self.real_academy:
                        self.fire()
            else:
                if action == 9 and abs(self._state[0] - self._state[2]) + abs(self._state[1] - self._state[3]) <= 15:
                    self.firing = 1
                    if self.real_academy:
                        self.game_controls.fire()
                else:
                    self.attacker.move(self.real_academy)
            x_a, y_a = self.attacker.get_obs()
            self.defender.move(x_a, y_a, self.real_academy)

            self.last_action = action
            self._state = self._get_obs()

        reward = self.get_reward()
        # this delay is necessary to sinc with the robot on real time
        self.save_step()
        if self.game_controls or self.human_mode:
            passed_time = time.time() - self.time_delay_last_time
            sleep_time = self.game_delay - passed_time
            if sleep_time >= 0:
                time.sleep(sleep_time)
            self.time_delay_last_time = time.time()

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(
                self._state, reward=reward)

    def get_reward(self):
        episode_speed_score = 0
        if self.game_controls:
            hit_score = (self.game_controls.hits_since_start - self.last_hits) * 100
            self.last_hits = self.game_controls.hits_since_start
            if self._episode_ended:
                episode_speed_score = 2401 - self.iterations
                print("________________________________________________________________")
                print("Game Score: {}".format(self.game_score))
                print("Game Time: {}".format(time.time() - self.game_start))
                print("Training Score: {}".format(self.whole_reward))
                print("________________________________________________________________")

            if hit_score > 10:
                print("HIT!!! REJOICEEEEEEEEEEEEEE")
                self.level += 1
                seconds = round(time.time() - self.last_hit_time)
                if seconds >= 20:
                    self.game_score += 20
                else:
                    self.game_score = self.game_score + 20 + (20 - seconds)

                self.game_controls.level = self.level
                self.hits += 1
                print("Level ended! game_score:{}".format(self.game_score))
                time.sleep(3)
                self.last_hit_time = time.time()
        else:
            hit_score = (self.hits - self.last_hits) * 100
            self.last_hits = self.hits
            if hit_score > 10:
                date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                print("{}: HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIT REJOOOOOOOOICEEEEEEEEEEEEE".format(date))
                self.reposition()
        # time discount every second

        distance = abs(self._state[0] - self._state[4]) + abs(self._state[1] - self._state[5])
        last_distance = abs(self.last_state[0] - self.last_state[4]) + abs(self.last_state[1] - self.last_state[5])
        reward = (hit_score + (last_distance - distance) / 10) + episode_speed_score
        self.whole_reward += reward
        print("Level: {}; Learn-Reward: {}, Game-Score: {}; Action:{}; Hits: {}; Speed: {} Hz; Ping: {}".format(
            self.level, round(self.whole_reward), self.game_score,
            self.last_action, self.last_hits, round(1 / (time.time() - self.last_measurement_time)), self._state[10]))
        self.last_measurement_time = time.time()
        return reward

    def fire(self):
        x_dist = abs(self._state[0] - self._state[4])
        y_dist = abs(self._state[1] - self._state[5])
        if x_dist <= 7 and y_dist <= 7:
            self.hits += 1
            self.level += 1

    def render(self, game_mode='human', close=False):
        """
        This function renders the current game state in the given mode.
        """
        if close:
            pygame.quit()
        else:
            if self.screen is None:
                self.pygame.init()
                self.screen = self.pygame.display.set_mode((self.window.get("x"), self.window.get("y")))
                self.attacker_py_image = self.pygame.image.load(self.attacker.image).convert_alpha()
                self.attacker_destination_py_image = self.pygame.image.load(self.attacker.target.image).convert_alpha()
                self.defender_py_image = self.pygame.image.load(self.defender.image).convert_alpha()
                self.background = self.pygame.image.load(self.background_image_path).convert()
                self.attacker.init_game_object(self.pygame.image.load(self.attacker.image).convert())
                self.defender.init_game_object(self.pygame.image.load(self.defender.image).convert())
                self.attacker.target.init_game_object(self.pygame.image.load(self.attacker.target.image).convert())
            # Set Images for the players
            clock = self.pygame.time.Clock()

            # blit the background
            self.screen.blit(self.background, (0, 0))

            # erase players
            self.screen.blit(self.background, self.defender.game_object.last_pos,
                             self.defender.game_object.last_pos)
            self.screen.blit(self.background, self.attacker.game_object.last_pos,
                             self.attacker.game_object.last_pos)
            self.screen.blit(self.background, self.attacker.target.game_object.last_pos,
                             self.attacker.target.game_object.last_pos)
            # display players in new state
            # x_a, y_a, x_a_dest, y_a_dest, x_d, y_d, x_dot_d, y_dot_d, self.firing, self.level, 0
            if self.javris_prediction:
                prediction = self.javris_prediction.predict_step(tf.convert_to_tensor(
                    [[[ self.last_action, self._state[0], self._state[1], self._state[2],self._state[3], self._state[4],self._state[5],self._state[6],self._state[7],self._state[8],
                       self._state[9], (self._state[10]+0.5)]]])).numpy()
                if self.level <= 7:
#                    shifter = (self.latencies[self.level-1] + self.game_controls.ping*1000)/1000
                     shifter = 1
                else:
                    shifter = 0
                #a_x_norm = ((prediction[0][0][0] - self._state[0])*shifter)+self._state[0]
                #a_y_norm = ((prediction[0][0][1] - self._state[1])*shifter)+self._state[1]
                #d_x_norm = ((prediction[0][0][2] - self._state[4])*shifter)+self._state[4]
                #d_y_norm = ((prediction[0][0][3] - self._state[5])*shifter)+self._state[5]

                a_x_norm = prediction[0][0][0]
                a_y_norm = prediction[0][0][1]
                d_x_norm = prediction[0][0][2]
                d_y_norm = prediction[0][0][3]

                #a_x_norm = (self._state[1]+prediction[0][0][0]*shifter)/2
                #a_y_norm = (self._state[1]+prediction[0][0][1]*shifter)/2
                #d_x_norm = (self._state[4]+prediction[0][0][2]*shifter)/2
                #d_y_norm = (self._state[5]+prediction[0][0][3]*shifter)/2

                a_x, a_y = get_pos_in_pixel({"x":a_x_norm, "y":a_y_norm}, self.defender_size, self.window.get("x"), self.window.get("y"))
                d_x, d_y = get_pos_in_pixel({"x": d_x_norm, "y":d_y_norm}, self.defender_size,self.window.get("x"), self.window.get("y"))
                ad_x, ad_y = get_pos_in_pixel(self.target.position, self.defender_size,self.window.get("x"), self.window.get("y"))

            else:
                d_x, d_y = get_pos_in_pixel(self.defender.position,self.defender_size, self.window.get("x"), self.window.get("y"))
                a_x, a_y = get_pos_in_pixel(self.attacker.position, self.defender_size,self.window.get("x"), self.window.get("y"))
                ad_x, ad_y = get_pos_in_pixel(self.target.position, self.defender_size,self.window.get("x"), self.window.get("y"))

            self.screen.blit(self.defender_py_image, (d_x, d_y))
            self.screen.blit(self.attacker_py_image, (a_x, a_y))
            self.screen.blit(self.attacker_destination_py_image, (ad_x, ad_y))
            for point in self.defender.ellipsis:
                p_x, p_y = get_pos_in_pixel(point,0, self.window.get("x"), self.window.get("y"))

                if self.defender.ellipsis[self.defender.attacker_possiton_index] == point:
                    self.pygame.gfxdraw.filled_circle(
                        self.screen,
                        round(p_x),
                        round(p_y),
                        5,
                        (255, 0, 0))
                elif self.defender.ellipsis[self.defender.goal_point] == point:
                    self.pygame.gfxdraw.filled_circle(
                        self.screen,
                        round(p_x),
                        round(p_y),
                        5,
                        (0, 255, 0))
                else:
                    self.pygame.gfxdraw.filled_circle(
                        self.screen,
                        round(p_x),
                        round(p_y),
                        5,
                        (100, 100, 100))

            self.pygame.display.update()
            return 0

    def close(self):
        if self.screen:
            self.viewer = None
            self.pygame.quit()

    def is_same_state(self):
        if (self.last_state is not None and self._state is not None) and len(self.last_state) > 1 and len(
                self._state) > 1 and not self.human_mode:
            if (self.last_state[0] == self._state[0] and self.last_state[1] == self._state[1]) or (
                    self.last_state[4] == self._state[4] and self.last_state[5] == self._state[5]):
                self.same_buffer += 1
            else:
                self.same_buffer = 0
            if self.same_buffer == 600:
                raise Exception("Defender or Attacker Script died...")

            #
# env = CatchMeIfYouCanEnv()
# time_step = env.reset()
# print(time_step)
# cumulative_reward = time_step.reward

# for _ in range(3):
#    time_step = env.step(np.array(0))
#    print(time_step)
#    cumulative_reward += time_step.reward
#
# print(time_step)
# cumulative_reward += time_step.reward
# print('Final Reward = ', cumulative_reward)
