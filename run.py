import torch
import torch.nn as nn
import model
import config
import tic_tac_toe
import connect_4
import othello
import mcts
import random
import copy
import argparse


new_model = None
original = None
backup = None
game_type = ''
positions = []


def get_board():
    if game_type[:3] == 'tic':
        return tic_tac_toe.TicTacToeBoard(int(game_type[-1]))
    elif game_type == 'connect-4':
        return connect_4.Connect4Board()
    else:
        return othello.OthelloBoard()


def play(self_play, human_test):
    iters = config.SELF_PLAYS if self_play else config.EVAL_ITERS
    wins = 0
    if human_test:
        players = (None, new_model)
    elif self_play:
        players = (new_model, new_model)
    else:
        players = (new_model, backup)

    for i in range(iters):
        if human_test:
            print('Enter a move ' + ('("0 0" is the top-left space).'
                  if game_type[:3] == 'tic' or game_type == 'othello'
                  else '("0" is the leftmost column).'))
        else:
            print('Starting {} game #{}...'.format(
                'self-play' if self_play else 'evaluation', i + 1))
        board = get_board()
        curr_positions = []
        player_1_first = random.random() > .5 or not human_test

        while not board.is_game_over():
            curr_model = players[player_1_first ^ board.turn]

            if curr_model:
                root = mcts.simulate(curr_model, board)
                if self_play:
                    curr_positions.append(
                        [copy.deepcopy(board), root.children])

                # Moves with the most visits are prioritized
                next_move = random.choices(
                    root.children, weights=list(map(
                        lambda child: child.visits, root.children)))[0].move
                board.push(next_move)
            else:
                print(board)
                move = input()
                try:
                    if game_type[:3] == 'tic' or game_type == 'othello':
                        move = move.split()
                        move = int(move[0]) * board.size + int(move[1])
                    else:
                        move = int(move)

                    if move in board.legal_moves:
                        board.push(move)
                    else:
                        print('ERROR: Illegal move.')
                except (ValueError, IndexError):
                    print('ERROR: Move cannot be parsed.')

        if human_test:
            print(board)
        winner = {'1-0': 1, '0-1': 0, '1/2-1/2': .5}[board.winner]
        print('Result of game is {}.'.format(board.winner))
        if self_play:
            for pos in curr_positions:
                pos.append(winner if pos[0].turn else 1 - winner)
            positions.extend(curr_positions)
        else:
            print('New model has {}.'.format(
                {1: 'WON', 0: 'LOST', .5: 'DRAWN'}[winner]))
            wins += winner

    return wins


def solve():
    board = get_board()
    # May be modified to evaluate a different position
    sequence = ((0, 0), (1, 1), (0, 1))
    for move in sequence:
        move = move[0] * board.size + move[1] \
            if game_type[:3] == 'tic' or game_type == 'othello' else move
        board.push(move)
    print(board)

    for i in range(10):
        root = mcts.simulate(new_model, board)
        next_move = random.choices(
            root.children, weights=list(map(
                lambda child: child.visits, root.children)))[0].move
        print('Predicted next move for {}: {}.'.format(
            'X' if board.turn else 'O',
            '({}, {})'.format(next_move // board.size, next_move % board.size)
            if game_type[:3] == 'tic' or game_type == 'othello'
            else next_move))


def get_y_policies(batch):
    Y_policies = torch.zeros([len(batch), batch[0][0].num_total_moves],
                             dtype=torch.float).cuda()
    for i, pos in enumerate(batch):
        visits_sum = sum(map(lambda child: child.visits, pos[1]))
        for j, child in enumerate(pos[1]):
            Y_policies[i, child.move] = child.visits / visits_sum
    return Y_policies


def train():
    print('Training new network...')
    global backup, positions
    backup = copy.deepcopy(new_model)
    new_model.train()

    positions = positions[-config.NUM_POSITIONS:]   # Discard older positions
    random.shuffle(positions)

    optimizer = torch.optim.Adam(new_model.parameters(),
                                 lr=config.LEARNING_RATE)
    for i in range(config.NUM_EPOCHS):
        epoch_loss = 0

        for j in range(0, len(positions), config.BATCH_SIZE):
            batch = positions[j:j+config.BATCH_SIZE]

            board_tensors = torch.stack([mcts.board_to_tensor(pos[0])
                                         .squeeze(0) for pos in batch]).cuda()

            X_policies, X_values = new_model(board_tensors)
            Y_policies = get_y_policies(batch)
            Y_values = torch.tensor([pos[2] for pos in batch],
                                    dtype=torch.float).cuda()

            loss = nn.MSELoss()(X_values, Y_values) \
                + torch.mean(Y_policies * -torch.log(X_policies))
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = round(
            epoch_loss.item() / -(-len(positions) // config.BATCH_SIZE), 4)
        print('Avg. batch loss for epoch #{}: {}.'.format(i + 1, avg_loss))


def evaluate(iter):
    global new_model
    wins = play(False, False)
    print('Score against previous network: {}/{}.'.format(
        wins, config.EVAL_ITERS))
    if wins >= config.EVAL_ITERS * .55:
        print('New network is better!')
    else:
        print('New network is not better. Restoring backup...')
        new_model = backup

    if iter % 5 == 0:
        file_name = 'model_iter_{}'.format(iter)
        print('Saving best network as "{}".'.format(file_name))
        with open(file_name, 'wb') as f:
            torch.save(new_model, f)


def final_test():
    # Run to confirm that the model has truly been learning

    print('Starting final evaluation...')
    global backup
    backup = original
    print('Score against original network: {}/{}.'.format(
        play(False, False), config.EVAL_ITERS))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', action='store', required=True)
    # Both arguments are required if one is entered
    parser.add_argument('--action', action='store')
    parser.add_argument('--model', action='store')
    args = parser.parse_args()

    return {
        'game': args.game,
        'play_against': args.action == 'play_against',
        'to_solve': args.action == 'solve',
        'model_file': args.model
    }


def main(game, play_against, to_solve, model_file):
    global new_model, original, game_type
    new_model = model.CNN(game).cuda()
    original = copy.deepcopy(new_model)
    game_type = game

    if play_against or to_solve:
        with open(model_file, 'rb') as f:
            new_model = torch.load(f)

        if to_solve:
            solve()
        else:
            print('Score against {}: {}/{}.'.format(
                model_file, play(True, True), config.EVAL_ITERS))
    else:
        for i in range(config.FULL_ITERS):
            print('Starting full iteration #{}...'.format(i + 1))
            play(True, False)
            train()
            evaluate(i + 1)
        final_test()


if __name__ == '__main__':
    main(**parse_args())
