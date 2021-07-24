import gym_game
from ..game_controls import GameControls
from .physics.physics import *
import math


class Robot:
    # [v_max] = should be given in % relative to the max spec velocity of Franka Emika specs
    # [a_max] = should be given in % relative to the max spec acceleration of Franka Emica specs
    # values from Franka Emika specs
    def __init__(self, image, height, x, y, x_min, x_max, y_min, y_max, v_max, a_max, step=2) -> None:
        self.step = step
        spec_v_max = 1.7
        spec_a_max = 13
        self.image = image
        self.height = height
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        # The max velocity of the robot in m/s
        self.v_max = v_max * spec_v_max
        # The max acceleration of the robot in m/s²
        self.a_max = a_max * spec_a_max
        self.a = 0
        self.v = 5
        self.position = {
            "x": x,
            "y": y
        }
        self.game_object = None
        self.last_position = self.position

    def get_obs(self):
        return self.position.values()

    def init_game_object(self, pygame_image):
        self.game_object = GameObject(pygame_image=pygame_image, height=self.height)

    def get_last_position(self):
        return self.last_position


class GameObject:
    def __init__(self, pygame_image, height):
        self.image = pygame_image
        self.pos = self.image.get_rect().move(0, height)
        self.last_pos = self.pos

    def __move__(self, pos_x, pos_y):
        self.last_pos = self.pos
        self.pos = self.pos.move(pos_x, pos_y)


class Attacker(Robot):
    def __init__(self, target, image, height, x, y, v_max, x_min, x_max, y_min, y_max, step,
                 game_controls: GameControls = None, a_max=1, javris_prediction=None):
        super().__init__(image=image, height=height, x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                         v_max=v_max, a_max=a_max, step=step)
        self.game_controls = game_controls
        self.target: Target = target
        self.x_latency_buffer = []
        self.y_latency_buffer = []

    def get_obs(self):
        if self.game_controls:
            self.position["x"] = self.game_controls.attacker_position[0]
            self.position["y"] = self.game_controls.attacker_position[1]
        return self.position.values()

    def move(self, real_academy, latency=1):
        dest_x, dest_y = self.target.position.values()
        if not real_academy:
            x = self.position["x"]
            y = self.position["y"]

            x_dir = 1 if dest_x - x > self.step else -1 if dest_x - x < -self.step else 0
            y_dir = 1 if dest_y - y > self.step else -1 if dest_y - y < -self.step else 0

            x += x_dir * self.step
            y += y_dir * self.step

            x = ensure_bounds(x, self.x_min, self.x_max)
            y = ensure_bounds(y, self.y_min, self.y_max)

            # Update Position
            self.position["x"] = x
            self.position["y"] = y
        else:
            self.position["x"] = self.game_controls.attacker_position[0]
            self.position["y"] = self.game_controls.attacker_position[1]
            if abs(self.position["x"] - dest_x) > 2 or abs(self.position["y"] - dest_y) > 2:
                self.game_controls.moveTo(position=[dest_x, dest_y])

        x_p, y_p = get_pos_in_pixel(self.position, self.height,self.x_max, self.y_max)
        if self.game_object:
            self.game_object.__move__(x_p, y_p)

        return self.position


class Target(Robot):
    def __init__(self, image, height, x, y, v_max, x_min, x_max, y_min, y_max, a_max=2, step=2):
        super().__init__(image=image, height=height, x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                         v_max=v_max, a_max=a_max)

    def move(self, action):
        self.last_position = self.position

        x = self.position["x"]
        y = self.position["y"]
        # move left
        if action == 1:
            x = x - self.step
            # move right
        elif action == 2:
            x = x + self.step
            # move up
        elif action == 3:
            y = y - self.step
        # move down
        elif action == 4:
            y = y + self.step
        # move left/up
        elif action == 5:
            x = x - self.step
            y = y - self.step
        # move left/down
        elif action == 6:
            x = x - self.step
            y = y + self.step
        # move right/down
        elif action == 7:
            x = x + self.step
            y = y + self.step
        # move right/up
        elif action == 8:
            x = x + self.step
            y = y - self.step

        # Boundaries

        x = ensure_bounds(x, self.x_min, self.x_max)
        y = ensure_bounds(y, self.y_min, self.y_max)

        # Update Position
        self.position["x"] = x
        self.position["y"] = y

        x_p, y_p = get_pos_in_pixel(self.position,self.height, self.x_max, self.y_max)
        if self.game_object:
            self.game_object.__move__(x_p, y_p)

        return self.position


class Defender(Robot):
    def __init__(self, image, height, x, y, x_dot, y_dot, x_min, x_max, y_min, y_max,
                 game_controls: GameControls = None, v_max=1, a_max=1):
        super().__init__(image=image, height=height, x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                         v_max=v_max, a_max=a_max)
        self.game_controls = game_controls
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.dist_to_attacker = []
        self.x_dir = 1.0
        self.y_dir = -1.0
        self.attacker_possiton_index = 0
        self.goal_point = 0
        self.points = 100
        self.distance_attacker_ellipse = 0
        self.a = (self.x_max - self.x_min) / 2
        self.b = (self.y_max - self.y_min) / 2
        self.centerx = self.x_min + self.a
        self.centery = self.y_min + self.b
        self.ellipsis = create_ellipsis(centerx=self.centerx, centery=self.centery, a=self.a, b=self.b,
                                        points=self.points)

    def get_obs(self):
        # TODO: Velocities
        if self.game_controls:
            self.position["x"] = self.game_controls.defender_position[0]
            self.position["y"] = self.game_controls.defender_position[1]
        return self.position.values()

    def move(self, a_x, a_y, real_academy=False):
        if real_academy:
            self.get_obs()
            self.attacker_possiton_index = get_attacker_possition_index_on_ellipse(ellipsis=self.ellipsis, a_x=a_x,
                                                                                   a_y=a_y)
            self.goal_point = int(self.attacker_possiton_index + (self.points / 2)) % self.points
        else:
            # change direction
            #  Some magic blur around the target position
            blur = 0.5
            step = 1
            # step = 0.05
            # blur = 0.5          
            # The Goal is the further most point in the ellipsis from the attacker.
            self.attacker_possiton_index = get_attacker_possition_index_on_ellipse(ellipsis=self.ellipsis, a_x=a_x,
                                                                                   a_y=a_y)
            self.goal_point = int(self.attacker_possiton_index + (self.points / 2)) % self.points
            # I dont know why but the defender just runs away in the Y direction. See Code random-walk.h
            # double goalx = std::get<0>(ellipse[detective_ellipse_distance_index]);
            # double goaly = std::get<1>(ellipse[goal]);
            x = self.position["x"]
            y = self.position["y"]

            goalx = self.ellipsis[self.goal_point].get("x")
            goaly = self.ellipsis[self.goal_point].get("y")
            x_dir = 1.0
            y_dir = 1.0
            if (goalx < x):
                x_dir = -1.0

            if (goalx > x):
                x_dir = +1.0

            if (-1 * blur <= (x - a_x) <= blur):
                x_dir = 1.0

            if (-1 * blur <= (a_x - x) <= blur):
                x_dir = -1.0

            if (goaly < y):
                y_dir = -1.0

            if (goaly > y):
                y_dir = 1.0

            if (-1 * blur <= (y - a_y) <= blur):
                y_dir = 1.0

            if (-1 * blur <= (a_y - y) <= blur):
                y_dir = -1.0

            x += x_dir * step
            y += y_dir * step

            # print("Defender: Direction ({}, {})".format(x_dir, y_dir))
            self.v = self.v_max if self.v > self.v_max else self.v
            self.v = self.v_min if self.v < -self.v_max else self.v
            x = ensure_bounds(x, self.x_min, self.x_max)
            y = ensure_bounds(y, self.y_min, self.y_max)

            # Update Position
            self.position["x"] = x
            self.position["y"] = y

        x_p, y_p = get_pos_in_pixel(self.position, self.height,self.x_max, self.y_max)
        if self.game_object:
            self.game_object.__move__(x_p, y_p)

        return self.position
