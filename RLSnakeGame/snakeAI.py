import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple


pygame.init()

font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 255, 100)
BACKGROUND = (50, 50, 50)

BLOCK_SIZE = 20
BASE_SPEED = 20
SPEED_MULTIPLIERS = [0.25, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000, 2000]

GAME_AREA_WIDTH = 640
GAME_AREA_HEIGHT = 480

SNAKE_SHAPE = 1  # 0 = circle, 1 = rectangle
SNAKE_COLOR_1 = GREEN1
SNAKE_COLOR_2 = GREEN2

REWARD_FOOD_DISTANCE = True  # positive reward for approaching food; negative reward for moving away from food
                             # if True, prevents snake from spinning in circles

CHECK_TRAPPED = False


class SnakeGameAI:
    def __init__(self, w=GAME_AREA_WIDTH, h=GAME_AREA_HEIGHT):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.speed_multiplier_index = 2 # set initial game speed to 1 X BASE_SPEED
        self.attempt_count = 1

        # init game state
        self.reset()

    def reset(self, record = 0):
        # init/reset game state
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
        # self.prev_food_distance = 1000


    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        self.prev_food_distance = self.get_food_distance()
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        # increase frame iteration
        self.frame_iteration += 1

        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    self.speed_multiplier_index += 1
                    self.speed_multiplier_index = self.speed_multiplier_index % len(SPEED_MULTIPLIERS)
                elif event.key == pygame.K_z:
                    self.speed_multiplier_index -= 1
                    self.speed_multiplier_index = self.speed_multiplier_index % len(SPEED_MULTIPLIERS)

        self.prev_food_distance = self.get_food_distance()

        # move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # check if game is over
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

        # place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            if REWARD_FOOD_DISTANCE:
                reward = self.compute_reward()
                # print("Reward: " + str(reward))
            self.snake.pop()


        # update ui and clock
        self._update_ui()
        self.clock.tick(BASE_SPEED * SPEED_MULTIPLIERS[self.speed_multiplier_index])

        # return reward, game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BACKGROUND)

        for pt in self.snake:
            if SNAKE_SHAPE == 0:
                pygame.draw.circle(self.display, SNAKE_COLOR_1, (pt.x + BLOCK_SIZE / 2, pt.y + BLOCK_SIZE / 2), BLOCK_SIZE / 2)
            else:
                pygame.draw.rect(self.display, SNAKE_COLOR_1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, SNAKE_COLOR_2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # display attempt number
        text = font.render("Attempt: " + str(self.attempt_count), True, WHITE)  # insert correct attempt number here
        self.display.blit(text, [0, 0])

        # display score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 25])

        # display speed multiplier
        text = font.render("Speed: " + str(SPEED_MULTIPLIERS[self.speed_multiplier_index]) + "x", True, WHITE)
        self.display.blit(text, [GAME_AREA_WIDTH / 2 - 70, 0])

        # display highscore
        text = font.render("Highscore: " + str(self.highscore), True, WHITE)
        self.display.blit(text, [GAME_AREA_WIDTH - 150, 0])

        pygame.display.flip()

    def _move(self, action):
        # determine snake direction based on current action
        # possible actions: [forward, right, left]
        clockwise_dir = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        dir_index = clockwise_dir.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise_dir[dir_index] # no change in direction
        elif np.array_equal(action, [0, 1, 0]):
            new_dir_index = (dir_index + 1) % 4
            new_dir = clockwise_dir[new_dir_index] # direction -> turn right
        else:
            # [0, 0, 1]
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
        # Compute reward based on food distance (reward for approaching food, punish for going away from food)
        food_distance = self.get_food_distance()
        # print("Food distance: " + str(food_distance) + " Prev food distance: " + str(self.prev_food_distance))
        if food_distance < self.prev_food_distance:
            return 2
        else:
            return -2


    def is_self_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def is_trapped(self):
        collision_points = 0
        # Simulate possible movements and check if the snake can move
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
            print("TRAPPED!")
            return True
        return False