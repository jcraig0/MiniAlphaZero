import torch
import torch.nn as nn
import model
import config
import tic_tac_toe
import mcts
import random
import copy
import argparse


model = model.TicTacToeCNN().cuda()
original = copy.deepcopy(model)
backup = None
positions = []


def play(self_play, human_test):
    iters = config.SELF_PLAYS if self_play else config.EVAL_ITERS
    wins = 0

    for i in range(iters):
        if not human_test:
            print('Starting {} game #{}...'.format(
                'self-play' if self_play else 'evaluation', i + 1))
        board = tic_tac_toe.TicTacToeBoard()
        curr_positions = []

        while not board.is_game_over():
            if human_test and board.turn:
                print(board)

                move = input().split()
                board.push((int(move[0]), int(move[1])))
            else:
                curr_model = model if self_play or board.turn else backup
                root = mcts.simulate(curr_model, board)
                if self_play:
                    curr_positions.append(
                        [copy.deepcopy(board), root.children])

                # Moves with the most visits are prioritized
                next_move = random.choices(
                    root.children, weights=list(map(
                        lambda child: child.visits, root.children)))[0].move
                board.push(next_move)

        if human_test:
            print(board)
        winner = {'1-0': 1, '0-1': 0, '1/2-1/2': .5}[board.winner]
        print('Result of game is {}.'.format(board.winner))
        if self_play:
            for pos in curr_positions:
                pos.append(winner if pos[0].turn else 1 - winner)
            positions.extend(curr_positions)
        else:
            wins += winner

    return wins


def solve():
    board = tic_tac_toe.TicTacToeBoard()
    # May be modified to evaluate a different position
    sequence = [(2, 1), (1, 1), (1, 2), (3, 3), (1, 3), (0, 0)]
    for move in sequence:
        board.push(move)
    print('\n' + str(board) + '\n')

    for i in range(10):
        root = mcts.simulate(model, board)
        next_move = random.choices(
            root.children, weights=list(map(
                lambda child: child.visits, root.children)))[0].move
        print('Predicted next move for {}: {}.'.format(
            'X' if board.turn else 'O', next_move))


def get_y_policies(batch):
    Y_policies = torch.zeros([len(batch), 16], dtype=torch.float).cuda()
    for i, pos in enumerate(batch):
        visits_sum = sum(map(lambda child: child.visits, pos[1]))
        for j, child in enumerate(pos[1]):
            Y_policies[i, child.move[0] * 4 + child.move[1]] = \
                child.visits / visits_sum
    return Y_policies


def train():
    print('Training new network...')
    global backup, positions
    backup = copy.deepcopy(model)
    model.train()

    positions = positions[-config.NUM_POSITIONS:]   # Discard older positions
    random.shuffle(positions)

    optimizer = torch.optim.Adam(model.parameters())

    for i in range(config.NUM_EPOCHS):
        epoch_loss = 0

        for j in range(0, len(positions), config.BATCH_SIZE):
            batch = positions[j:j+config.BATCH_SIZE]

            board_tensors = torch.stack([mcts.board_to_tensor(pos[0])
                                         .squeeze(0) for pos in batch]).cuda()

            X_policies, X_values = model(board_tensors)
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
    global model
    wins = play(False, False)
    print('Score against previous network: {}/{}.'.format(
        wins, config.EVAL_ITERS))
    if wins >= config.EVAL_ITERS * .55:
        print('New network is better!')
    else:
        print('New network is not better. Restoring backup...')
        model = backup

    if iter % 5 == 0:
        file_name = 'model_iter_{}'.format(iter)
        print("Saving best network as '{}'.".format(file_name))
        with open(file_name, 'wb') as f:
            torch.save(model, f)


def final_test():
    # Run to confirm that the model has truly learned the game

    print('Starting final evaluation...')
    global backup
    backup = original
    print('Score against original network: {}/{}.'.format(
        play(False, False), config.EVAL_ITERS))


def parse_args():
    parser = argparse.ArgumentParser()
    # Both arguments are required if one is entered
    parser.add_argument('--action', action='store')
    parser.add_argument('--model', action='store')
    args = parser.parse_args()

    return {
        'play_against': args.action == 'play_against',
        'to_solve': args.action == 'solve',
        'model_file': args.model
    }


def main(play_against, to_solve, model_file):
    global model
    if play_against or to_solve:
        with open(model_file, 'rb') as f:
            model = torch.load(f)

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
