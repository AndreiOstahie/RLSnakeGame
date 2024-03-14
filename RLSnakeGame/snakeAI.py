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
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
BASE_SPEED = 20
SPEED_MULTIPLIERS = [0.25, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000, 2000]


class SnakeGameAI:
    def __init__(self, w=640, h=480):
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


    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        # increase frame iteration
        self.frame_iteration += 1

        # 1. collect user input
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


        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(BASE_SPEED * SPEED_MULTIPLIERS[self.speed_multiplier_index])
        # 6. return game over and score
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
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # display attempt number
        text = font.render("Attempt: " + str(self.attempt_count), True, WHITE) # insert correct attempt number here
        self.display.blit(text, [0, 0])

        # display score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 25])

        # display speed multiplier
        text = font.render("Speed: " + str(SPEED_MULTIPLIERS[self.speed_multiplier_index]) + "x", True, WHITE)
        self.display.blit(text, [0, 50])

        # display highscore
        text = font.render("Highscore: " + str(self.highscore), True, WHITE)
        self.display.blit(text, [0, 75])

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
