class Connect4Board():
    def __init__(self):
        self.grid = [[' '] * 7 for i in range(6)]
        self.turn = True
        self.winner = ''
        self.legal_moves = [i for i in range(7)]
        self.num_all_moves = 7

    def is_game_over(self):
        for i in range(6):
            for j in range(7):
                if (self.grid[i][j] != ' ' and
                    # Check for a row win
                    (self.grid[i][j] == self.grid[i+1][j] and
                     self.grid[i][j] == self.grid[i+2][j] and
                     self.grid[i][j] == self.grid[i+3][j]) or
                    # Check for a column win
                    (self.grid[i][j] == self.grid[i][j+1] and
                     self.grid[i][j] == self.grid[i][j+2] and
                     self.grid[i][j] == self.grid[i][j+3]) or
                    # Check for a diagonal win
                    (self.grid[i][j] == self.grid[i+1][j+1] and
                     self.grid[i][j] == self.grid[i+2][j+2] and
                     self.grid[i][j] == self.grid[i+3][j+3])):
                    return True
        # Check if the board is full
        if all([self.grid[i][j] != ' ' for i in range(7) for j in range(6)]):
            self.winner = '1/2-1/2'
            return True

    def push(self, move):
        row_idx = max([i for i in range(6) if self.grid[i][move] == ' '])
        self.grid[row_idx][move] = 'X' if self.turn else 'O'
        self.legal_moves.remove(move)
        self.turn = False if self.turn else True

    def __str__(self):
        return '\n' + '\n'.join(
            [' '.join(['_' if i == ' ' else i for i in row])
             for row in self.grid]) + '\n'
