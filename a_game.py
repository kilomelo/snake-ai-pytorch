import random
from enum import Enum
from collections import namedtuple
import numpy as np

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

Point = namedtuple('Point', 'x, y')

class SnakeGame:
    def __init__(self, width, height):
        self.map = np.zeros((width, height))
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        # init game state
        self.map[:, :] = 0
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            # Point(self.head.x - 2, self.head.y)
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

        reward = 0
        distance_pre = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)

        game_over = self.frame_iteration > 100*len(self.snake)

        # 2. move
        if not game_over:
            game_over = game_over or self._move(action) # update the head

        if game_over:
            reward -= 10
            return reward, game_over, self.score

        self.snake.insert(0, self.head)

        distance_after = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)

        if distance_pre > distance_after:
            reward += 5

        self.map[self.head.x, self.head.y] = 1
        
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
        else:
            tail = self.snake.pop()
            self.map[tail.x, tail.y] = 0

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
    
    def _move(self, action):
        new_dir = Direction(action.argmax())
        # print('_move, action:', action, 'new_dir:', new_dir)

        if new_dir == Direction.DOWN and self.direction == Direction.UP:
            new_dir = Direction.UP
        if new_dir == Direction.UP and self.direction == Direction.DOWN:
            new_dir = Direction.DOWN
        if new_dir == Direction.LEFT and self.direction == Direction.RIGHT:
            new_dir = Direction.RIGHT
        if new_dir == Direction.RIGHT and self.direction == Direction.LEFT:
            new_dir = Direction.LEFT

        self.map[self.head.x, self.head.y] = 2

        x = self.head.x
        y = self.head.y
        if new_dir == Direction.DOWN:
            y += 1
        if new_dir == Direction.UP:
            y -= 1
        if new_dir == Direction.LEFT:
            x -= 1
        if new_dir == Direction.RIGHT:
            x += 1

        new_head = Point(x, y)
        # 判断是否gameover
        if self.is_collision(new_head):
            return True
        
        self.head = new_head
        self.direction = new_dir
        return False
    
    def print(self):
        for h in range(self.height + 2):
            for w in range(self.width + 2):
                p = Point(w - 1, h - 1)
                if w == 0 or w == self.width + 1 or h == 0 or h == self.height + 1:
                    print('#', end = "\n" if w == self.width + 1 else "")
                elif self.map[p.x, p.y] == 1:
                    print('@', end="")
                elif self.map[p.x, p.y] == 2:
                    print('O', end="")
                elif self.map[p.x, p.y] == 3:
                    print('*', end="")
                else:
                    print(" ", end="")