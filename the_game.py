import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 80
SPEED = 500

class SnakeGame:
    def __init__(self, width, height):
        self.map = np.zeros((width, height))
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width * BLOCK_SIZE, self.height * BLOCK_SIZE))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.map[:, :] = 0
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y)
        ]
        self.map[self.head.x, self.head.y] = 1
        for body in self.snake[1:]:
            self.map[body.x, body.y] = 2
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, self.width - 1) 
        y = random.randint(0, self.height - 1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        else:
            self.map[self.food.x, self.food.y] = 3

    def play_step(self, action):
        self.frame_iteration += 1
        # wait_for_continue = True
        # while wait_for_continue:
            # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                # if event.type == pygame.KEYDOWN:
                    # if event.key == pygame.K_SPACE:
                        # wait_for_continue = False

        reward = 0
        distance_pre = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        distance_after = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)
        if distance_pre > distance_after:
            reward += 1

        # 3. check if game over
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        self.map[self.head.x, self.head.y] = 1
        
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            tail = self.snake.pop()
            self.map[tail.x, tail.y] = 0

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x >= self.width or pt.x < 0 or pt.y >= self.height or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False
    
    # def render(self):
    #     for h in range(self.height + 2):
    #         for w in range(self.width + 2):
    #             p = Point(w - 1, h - 1)
    #             if w == 0 or w == self.width + 1 or h == 0 or h == self.height + 1:
    #                 print(WALL, end = "\n" if w == self.width + 1 else "")
    #             elif self.map[p.x, p.y] == 1:
    #                 print(HEAD, end="")
    #             elif self.map[p.x, p.y] == 2:
    #                 print(BODY, end="")
    #             elif self.map[p.x, p.y] == 3:
    #                 print(FOOD, end="")
    #             else:
    #                 print(" ", end="")

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x * BLOCK_SIZE+4, pt.y * BLOCK_SIZE+4, BLOCK_SIZE-8, BLOCK_SIZE-8))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        self.map[x, y] = 2
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)

class SimpleGame:
    def __init__(self, width, height):
        pass
    def reset(self):
        pass
    def _place_food(self):
        pass
    def play_step(self, action):
        pass
    def is_collision(self, pt=None):
        pass
    def _update_ui(self):
        pass
    def _move(self, action):
        pass