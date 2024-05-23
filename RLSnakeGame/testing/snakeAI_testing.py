import random
import numpy as np
from enum import Enum
from collections import namedtuple

font = None  # No need for font in headless mode

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
BASE_SPEED = 20
SPEED_MULTIPLIERS = [0.25, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000, 2000]

GAME_AREA_WIDTH = 640
GAME_AREA_HEIGHT = 480

REWARD_FOOD_DISTANCE = False  # positive reward for approaching food; negative reward for moving away from food
                              # if True, prevents snake from spinning in circles

CHECK_TRAPPED = False

class SnakeGameAI:
    def __init__(self, w=GAME_AREA_WIDTH, h=GAME_AREA_HEIGHT):
        self.w = w
        self.h = h
        self.speed_multiplier_index = 10 # set initial game speed to 1 X BASE_SPEED
        self.attempt_count = 1
        self.reset()

    def reset(self, record = 0):
        self.attempt_count += 1
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.highscore = record

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        self.prev_food_distance = self.get_food_distance()
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        self.prev_food_distance = self.get_food_distance()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        if CHECK_TRAPPED:
            if self.is_trapped():
                game_over = True
                reward = -50
                return reward, game_over, self.score

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            if REWARD_FOOD_DISTANCE:
                reward = self.compute_reward()
            self.snake.pop()

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clockwise_dir = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        dir_index = clockwise_dir.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise_dir[dir_index] # no change in direction
        elif np.array_equal(action, [0, 1, 0]):
            new_dir_index = (dir_index + 1) % 4
            new_dir = clockwise_dir[new_dir_index] # direction -> turn right
        else:
            new_dir_index = (dir_index - 1) % 4
            new_dir = clockwise_dir[new_dir_index]  # direction -> turn left

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_food_distance(self):
        return abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)

    def compute_reward(self):
        food_distance = self.get_food_distance()
        if food_distance < self.prev_food_distance:
            return 2
        else:
            return -2

    def is_self_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt in self.snake[1:]:
            return True
        return False

    def is_trapped(self):
        collision_points = 0
        directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        for direction in directions:
            x, y = self.head
            if direction == Direction.RIGHT:
                x += BLOCK_SIZE
            elif direction == Direction.LEFT:
                x -= BLOCK_SIZE
            elif direction == Direction.DOWN:
                y += BLOCK_SIZE
            elif direction == Direction.UP:
                y -= BLOCK_SIZE

            if self.is_self_collision(Point(x, y)):
                collision_points += 1

        if collision_points == 4:
            return True
        return False
