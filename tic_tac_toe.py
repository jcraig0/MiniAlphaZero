class TicTacToeBoard():
    def __init__(self, size):
        self.grid = [[' '] * size for i in range(size)]
        self.turn = True
        self.winner = ''
        self.legal_moves = [i for i in range(size * size)]
        self.num_total_moves = size * size
        self.size = size

    def is_game_over(self):
        # Check for a row win
        for row in self.grid:
            if all([j == 'X' for j in row]) or all([j == 'O' for j in row]):
                self.winner = '0-1' if self.turn else '1-0'
                return True
        # Check for a column win
        for row in zip(*self.grid):
            if all([j == 'X' for j in row]) or all([j == 'O' for j in row]):
                self.winner = '0-1' if self.turn else '1-0'
                return True
        # Check for a diagonal win
        if (all([self.grid[i][i] == 'X' for i in range(self.size)])
           or all([self.grid[i][i] == 'O' for i in range(self.size)])
           or all([self.grid[i][self.size - 1 - i] == 'X'
                   for i in range(self.size)])
           or all([self.grid[i][self.size - 1 - i] == 'O'
                   for i in range(self.size)])):
            self.winner = '0-1' if self.turn else '1-0'
            return True
        # Check if the board is full
        if all([self.grid[i][j] != ' '
                for i in range(self.size) for j in range(self.size)]):
            self.winner = '1/2-1/2'
            return True

    def push(self, move):
        self.grid[move // self.size][move % self.size] = \
            'X' if self.turn else 'O'
        self.legal_moves.remove(move)
        self.turn = False if self.turn else True

    def __str__(self):
        return '\n' + '\n'.join(
            [' '.join(['_' if i == ' ' else i for i in row])
             for row in self.grid]) + '\n'
