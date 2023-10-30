import pygame
import torch
import random
import numpy as np
from collections import deque
from the_game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.01

class Agent:

    def __init__(self, model):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = game.map.flatten()
        # print(state)
        return state
        # return game.map.flatten()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(0.3, 0.7 - self.n_games * 0.0005)
        final_move = [0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGame(4, 3)
    model = Linear_QNet(game.width * game.height, 256, 3)
    model.load()
    agent = Agent(model)

    quit_train = False
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score, user_quit = game.play_step(final_move)
        quit_train = quit_train or user_quit
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', agent.epsilon)

            plot_scores.append(score)
            total_score += score
            # mean_score = total_score / agent.n_games
            mean_score = average_of_last_n_items(plot_scores, 100)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            if user_quit:
                agent.model.save()
                pygame.quit()
                quit()

def average_of_last_n_items(lst, n):
    # 边界情况：当n为0或负数时，返回None
    if n <= 0:
        return None
    
    # 边界情况：当列表为空时，返回None
    if not lst:
        return None

    # 如果n大于列表的长度，使用整个列表
    n = min(n, len(lst))
    
    # 使用切片获取末尾n项，并计算平均值
    return sum(lst[-n:]) / n

if __name__ == '__main__':
    train()