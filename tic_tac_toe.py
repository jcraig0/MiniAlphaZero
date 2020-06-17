class TicTacToeBoard():
    def __init__(self):
        self.grid = [[' '] * 4, [' '] * 4, [' '] * 4, [' '] * 4]
        self.fullmove_number = 1
        self.turn = True
        self.winner = ''
        self.legal_moves = [i for i in range(16)]

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
        if (all([self.grid[i][i] == 'X' for i in range(4)])
           or all([self.grid[i][i] == 'O' for i in range(4)])
           or all([self.grid[i][3 - i] == 'X' for i in range(4)])
           or all([self.grid[i][3 - i] == 'O' for i in range(4)])):
            self.winner = '0-1' if self.turn else '1-0'
            return True
        # Check if the board is full
        if all([self.grid[i][j] != ' ' for i in range(4) for j in range(4)]):
            self.winner = '1/2-1/2'
            return True

    def push(self, move):
        self.grid[move // 4][move % 4] = 'X' if self.turn else 'O'
        self.legal_moves.remove(move)

        self.turn = False if self.turn else True
        if not self.turn:
            self.fullmove_number += 1

    def __str__(self):
        return '\n' + '\n'.join(
            [' '.join(['_' if i == ' ' else i for i in row])
             for row in self.grid]) + '\n'
