class OthelloBoard():
    def __init__(self):
        self.grid = [[' '] * 8 for i in range(8)]
        self.grid[3][4] = self.grid[4][3] = 'X'
        self.grid[3][3] = self.grid[4][4] = 'O'
        self.turn = True
        self.winner = ''
        self.pieces = ('O', 'X')
        self.directions = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
                           (1, 1), (1, 0), (1, -1), (0, -1))
        self.legal_moves = self.get_legal_moves(self.turn)
        self.num_total_moves = 65
        self.size = 8

    def get_legal_moves(self, turn):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.grid[i][j] == self.pieces[turn]:
                    for dir in self.directions:
                        span = 1
                        while True:
                            new_i, new_j = i+dir[0]*span, j+dir[1]*span
                            if new_i < 0 or new_i > 7 or \
                               new_j < 0 or new_j > 7:
                                break
                            space = self.grid[new_i][new_j]
                            # If opponent's line of pieces continues
                            if space == self.pieces[1-turn]:
                                span += 1
                            else:
                                # If current player can place piece
                                # at end of the line
                                if space == ' ' and span != 1:
                                    moves.append(new_i * 8 + new_j)
                                break
        return set(moves) if moves else [64]

    def is_game_over(self):
        if self.legal_moves == [64] and \
           self.get_legal_moves(not self.turn) == [64]:
            num_X = sum([i == 'X' for row in self.grid for i in row])
            num_O = sum([i == 'O' for row in self.grid for i in row])
            if num_X > num_O:
                self.winner = '1-0'
            elif num_O > num_X:
                self.winner = '0-1'
            else:
                self.winner = '1/2-1/2'
            return True

    def flip_pieces(self, i, j):
        for dir in self.directions:
            places = []
            span = 1
            while True:
                new_i, new_j = i+dir[0]*span, j+dir[1]*span
                if new_i < 0 or new_i > 7 or new_j < 0 or new_j > 7:
                    break
                # If opponent's line of pieces continues
                if self.grid[new_i][new_j] == self.pieces[1-self.turn]:
                    places.append((new_i, new_j))
                    span += 1
                else:
                    # If current player's piece at end of the line
                    if self.grid[new_i][new_j] == self.pieces[self.turn]:
                        for place in places:
                            self.grid[place[0]][place[1]] = \
                                self.pieces[self.turn]
                    break

    def push(self, move):
        if move != 64:
            i, j = move // 8, move % 8
            self.grid[i][j] = self.pieces[self.turn]
            self.flip_pieces(i, j)
        self.turn = False if self.turn else True
        self.legal_moves = self.get_legal_moves(self.turn)

    def __str__(self):
        return '\n' + '\n'.join(
            [' '.join(['_' if i == ' ' else i for i in row])
             for row in self.grid]) + '\n'
