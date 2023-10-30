import pygame
import torch
import random
import numpy as np
from collections import deque
from a_game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 500
GAMMA = 0.9 # discount rate
MAX_EPSILON = 1
MIN_EPSILON = 0.01
EPSILON_DICREASE_RATE = 0.001

LR = 0.01

class Agent:

    def __init__(self, model):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = model
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)

    def get_state(self, game):
        state = game.map.flatten()
        return state
    
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
        self.epsilon = max(MIN_EPSILON, MAX_EPSILON - self.n_games * EPSILON_DICREASE_RATE)
        final_move = np.array([0,0,0,0])
        if random.random() < self.epsilon:
        # if False:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


FRAME_RATE = 1
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 80

def train():
    width = 4
    height = 3
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGame(width, height)
    model = Linear_QNet(game.width * game.height, 64, 4)
    # model.load()
    agent = Agent(model)

    pygame.init()
    display = pygame.display.set_mode((width * BLOCK_SIZE, height * BLOCK_SIZE))
    pygame.display.set_caption('Snake')
    clock = pygame.time.Clock()
    font = pygame.font.Font('arial.ttf', 25)
    # 是否观察游戏过程
    observe = False
    while True:
        if observe:
            display.fill(BLACK)
            for (x, y), value in np.ndenumerate(game.map):
                if value == 1:
                    pygame.draw.rect(display, BLUE1, pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                elif value == 2:
                    pygame.draw.rect(display, BLUE2, pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                elif value == 3:
                    pygame.draw.rect(display, RED, pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

            text = font.render("Score: " + str(game.score), True, WHITE)
            display.blit(text, [0, 0])
            pygame.display.flip()
            clock.tick(FRAME_RATE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # agent.model.save()
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    observe = not observe

        # get old state
        state_old = agent.get_state(game)
        # print('state_old:', state_old)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        # print('reward:', reward)
        # print('state_new:', state_new)
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
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', agent.epsilon)

            plot_scores.append(score)
            total_score += score
            # mean_score = total_score / agent.n_games
            mean_score = average_of_last_n_items(plot_scores, 100)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

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