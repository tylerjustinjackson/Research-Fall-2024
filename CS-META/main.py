import numpy as np
import random
from typing import Tuple

BOARD_SIZE = 25
EMPTY = 0
SNAKE = 1
FOOD = 2


class Snake:

    # this code is final
    def __init__(self, size: int = BOARD_SIZE) -> None:

        self.size = size
        self.board = self._initialize_board()
        self.snake = []
        self.direction = (0, 1)  # Initial direction: right
        self._initialize_snake()
        self._place_food()

    # this code is final
    def _initialize_board(self) -> np.ndarray:
        """make the game board with empty cells"""
        return np.zeros((self.size, self.size), dtype=int)

    # can be changed
    def _initialize_snake(self, initial_length: int = 3) -> None:
        """places the snake in the center of the board"""
        center = self.size // 2

        for s in range(initial_length):
            self.board[center, center - s] = SNAKE
            self.snake.append((center, center - s))

    # This code is final
    def _place_food(self) -> None:
        """place food in a random empty cell on the board"""
        empty_cells = np.argwhere(self.board == EMPTY)

        if len(empty_cells) > 0:
            food_position = empty_cells[random.choice(range(len(empty_cells)))]
            self.board[food_position[0], food_position[1]] = FOOD

    # This code is final
    def print_board(self) -> None:
        """Print the game board."""
        symbols = {EMPTY: ".", SNAKE: "S", FOOD: "F"}

        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print()

    # this is updating the board, needs to be asjusted if needed
    def move_snake(self) -> None:
        """shifts the snake to whatever the new direction is"""
        head_y, head_x = self.snake[-1]
        move_y, move_x = self.direction
        new_head_y, new_head_x = head_y + move_y, head_x + move_x

        # see if the new position is within board boundaries
        if 0 <= new_head_y < self.size and 0 <= new_head_x < self.size:
            if (new_head_y, new_head_x) in self.snake:
                print("Snake ran into itself, game over")
                return

            # moves snake
            self.snake.append((new_head_y, new_head_x))
            self.board[new_head_y, new_head_x] = SNAKE

            # sees if food is eaten
            if self.board[new_head_y, new_head_x] == FOOD:
                self._place_food()  # Place new food
            else:
                tail_y, tail_x = self.snake.pop(0)
                self.board[tail_y, tail_x] = EMPTY
        else:
            print("Snake went out of bounds, game over")

    # needs to be updated
    def set_direction(self, direction: Tuple[int, int]) -> None:
        """sets the direction of the snake."""
        # Avoid reversing the snake direction
        if (direction[0], direction[1]) != (-self.direction[0], -self.direction[1]):
            self.direction = direction


# place holder
def main() -> None:
    game = Snake()

    game.print_board()

    # Example moves
    game.set_direction((0, 0))
    game.move_snake()

    print("Board after move:")
    game.print_board()


if __name__ == "__main__":
    main()


# NOTE:  This pulls from "Automate the Boring Stuff with Python" by Al Sweigart
# NOTE: This uses PEP8
