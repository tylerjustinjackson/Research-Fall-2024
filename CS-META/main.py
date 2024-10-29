import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, List
import matplotlib.pyplot as plt

BOARD_SIZE = 25
EMPTY = 0
SNAKE = 1
FOOD = 2
INITIAL_LENGTH = 3
EPISODES = 10000
MAX_MEMORY = 2000
BATCH_SIZE = 256
GAMMA = 0.99


class Snake:
    def __init__(self, size: int = BOARD_SIZE) -> None:
        self.size = size
        self.board = self._initialize_board()
        self.snake = []
        self.direction = (0, 1)
        self.score = 0
        self._initialize_snake()
        self._place_food()

    def _initialize_board(self) -> np.ndarray:
        return np.zeros((self.size, self.size), dtype=int)

    def _initialize_snake(self) -> None:
        center = self.size // 2
        for s in range(INITIAL_LENGTH):
            self.board[center, center - s] = SNAKE
            self.snake.append((center, center - s))

    def _place_food(self) -> None:
        empty_cells = np.argwhere(self.board == EMPTY)
        if len(empty_cells) > 0:
            food_position = empty_cells[random.choice(range(len(empty_cells)))]
            self.board[food_position[0], food_position[1]] = FOOD

    def reset(self) -> None:
        self.board = self._initialize_board()
        self.snake = []
        self.direction = (0, 1)
        self.score = 0
        self._initialize_snake()
        self._place_food()

    def print_board(self) -> None:
        symbols = {EMPTY: ".", SNAKE: "S", FOOD: "F"}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print()

    def move_snake(self) -> Tuple[bool, int]:
        head_y, head_x = self.snake[-1]
        move_y, move_x = self.direction
        new_head_y, new_head_x = head_y + move_y, head_x + move_x

        if not (0 <= new_head_y < self.size and 0 <= new_head_x < self.size):
            return True, -10

        if (new_head_y, new_head_x) in self.snake:
            return True, -10

        self.snake.append((new_head_y, new_head_x))
        self.board[new_head_y, new_head_x] = SNAKE

        if self.board[new_head_y, new_head_x] == FOOD:
            self.score += 1
            self._place_food()
            return False, 20
        else:
            tail_y, tail_x = self.snake.pop(0)
            self.board[tail_y, tail_x] = EMPTY
            return False, -1

    def set_direction(self, direction: Tuple[int, int]) -> None:
        if (direction[0], direction[1]) != (-self.direction[0], -self.direction[1]):
            self.direction = direction

    def get_state(self) -> np.ndarray:
        return self.board.flatten()


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999995

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return np.argmax(q_values.detach().numpy())

    def replay(self) -> None:
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += GAMMA * np.max(
                    self.model(torch.FloatTensor(next_state)).detach().numpy()
                )

            target_f = self.model(torch.FloatTensor(state)).detach()
            target_f[action] = target
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(torch.FloatTensor(state))
            loss = self.loss_fn(output, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def plot_board(board):
    plt.imshow(board, cmap="gray", vmin=0, vmax=2)
    plt.xticks([])  # Hide x ticks
    plt.yticks([])  # Hide y ticks
    plt.show()


def main():
    game = Snake()
    agent = Agent(input_dim=BOARD_SIZE * BOARD_SIZE, output_dim=4)

    total_rewards = []

    for episode in range(EPISODES):
        state = game.get_state()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            direction = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            game.set_direction(direction[action])
            done, reward = game.move_snake()
            next_state = game.get_state()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            # plot_board(game.board)

        agent.replay()
        total_rewards.append(total_reward)
        game.reset()
        print(
            f"Episode {episode + 1}/{EPISODES}, Total reward: {total_reward}, Epsilon: {agent.epsilon:.2f}"
        )

    model_save_path = "model.pth"
    torch.save(agent.model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    with open("learning_progress.txt", "w") as f:
        for i, reward in enumerate(total_rewards):
            f.write(f"Episode {i + 1}: Total Reward: {reward}\n")

    print("Learning progress saved to learning_progress.txt")


if __name__ == "__main__":
    main()
