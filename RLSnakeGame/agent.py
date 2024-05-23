import torch
import random
import numpy as np
from snakeAI import SnakeGameAI, Point, Direction, BLOCK_SIZE
from collections import deque
from model import Linear_Qnet, QTrainer
from plothelper import plot, plot_thread

import multiprocessing

from threading import Thread

MAX_MEMORY = 100_000  # deque max memory size
BATCH_SIZE = 1000  # memory sample (min) size

LR = 0.001  # learning rate
DISCOUNT_RATE = 0.9  # discount rate (val < 1 - modify val for different results)
EXPLORATION_VAL = 75  # decision randomness = 0% after EXPLORATION_VAL attempts
HIDDEN_SIZE = 256  # neural network hidden layer size


MULTIPROC = False


MAX_ATTEMPTS = 200  # max number of attempts; set to 0 to disable


class Agent:
    def __init__(self):
        self.n_games = 0  # number of games
        self.epsilon = 0  # randomness param
        self.gamma = DISCOUNT_RATE  # discount rate (val < 1 - modify val for different results)
        self.memory = deque(maxlen=MAX_MEMORY)  # removes elements by using popleft() when max length is exceeded

        self.model = Linear_Qnet(11, HIDDEN_SIZE, 3)  # 11 possible states, 3 possible actions, HIDDEN_SIZE hidden layer size
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)



    def get_state(self, game):
        # 11 Possible states:
        # [ danger_forward, danger_right, danger_left,
        #   dir_left, dir_right, dir_up, dir_down,
        #   food_left, food_right, food_up, food_down ]
        # -- 0 if false, 1 if true --

        head = game.snake[0]
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [
            # danger_forward
            (dir_right and game.is_collision(point_right)) or
            (dir_left and game.is_collision(point_left)) or
            (dir_up and game.is_collision(point_up)) or
            (dir_down and game.is_collision(point_down)),

            # danger_right
            (dir_right and game.is_collision(point_down)) or
            (dir_left and game.is_collision(point_up)) or
            (dir_up and game.is_collision(point_right)) or
            (dir_down and game.is_collision(point_left)),

            # danger_left
            (dir_right and game.is_collision(point_up)) or
            (dir_left and game.is_collision(point_down)) or
            (dir_up and game.is_collision(point_left)) or
            (dir_down and game.is_collision(point_right)),

            # move direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # food position
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)



    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            memory_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            memory_sample = self.memory

        sample_states, sample_actions, sample_rewards, sample_next_states, sample_game_overs = zip(*memory_sample)  # extract and group items of the same type together
        self.trainer.train_step(sample_states, sample_actions, sample_rewards, sample_next_states, sample_game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff between exploration and exploitation
        self.epsilon = EXPLORATION_VAL - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # perform random action
            move = random.randint(0, 2)
            action[move] = 1
        else:
            # perform action based on the model
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)  # predict next action
            move = torch.argmax(prediction).item()  # get max argument from prediction and convert it from tensor to item (value)
            action[move] = 1

        return action

def train():
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Stop execution at final attempt
        if 0 < MAX_ATTEMPTS <= agent.n_games:
            while True:
                continue

        # get current state
        state = agent.get_state(game)

        # get action
        action = agent.get_action(state)

        # perform move and get new state
        reward, game_over, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state, action, reward, state_new, game_over)

        # remember
        agent.remember(state, action, reward, state_new, game_over)

        if game_over:
            # train long memory, plot results
            if score > record:
                record = score

            game.reset(record)
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.n_games
            plot_avg_scores.append(avg_score)

            plot(plot_scores, plot_avg_scores)

def train_multiproc(queue):
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get current state
        state = agent.get_state(game)

        # get action
        action = agent.get_action(state)

        # perform move and get new state
        reward, game_over, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state, action, reward, state_new, game_over)

        # remember
        agent.remember(state, action, reward, state_new, game_over)

        if game_over:
            # train long memory, plot results
            if score > record:
                record = score

            game.reset(record)
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.n_games
            plot_avg_scores.append(avg_score)

            queue.put((plot_scores, plot_avg_scores))


if __name__ == '__main__':
    if MULTIPROC:
        queue = multiprocessing.Queue()

        game_process = multiprocessing.Process(target=train_multiproc, args=(queue,))
        plot_process = multiprocessing.Process(target=plot_thread, args=(queue,))

        game_process.start()
        plot_process.start()

        game_process.join()
        plot_process.join()
    else:
        train()