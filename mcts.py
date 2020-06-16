import torch
import config
import math
import copy


class Node:
    def __init__(self):
        self.wins = 0
        self.visits = 0
        self.probs = None
        self.move = None
        self.children = []
        self.parent = None


def board_to_tensor(board):
    tensor = torch.zeros([1, 1, 4, 4], dtype=torch.float).cuda()
    for i in range(4):
        for j in range(4):
            value = {'X': 1, 'O': 0, ' ': .5}[board.grid[i][j]]
            tensor[0, 0, i, j] = value if board.turn else 1 - value
    return tensor


def ucb(curr_node, child):
    prob = curr_node.probs[child.move[0] * 4 + child.move[1]]
    if child.visits:
        # "-1 + 2 *" projects the range [0, 1] to [-1, 1]
        return -1 + 2 * child.wins / child.visits + prob \
            * math.sqrt(curr_node.visits) / (child.visits + 1)
    else:
        return prob * math.sqrt(curr_node.visits + 1e-8)


def simulate(model, board):
    model.eval()
    root = Node()

    for i in range(config.MCTS_SIMS):
        board_copy = copy.deepcopy(board)
        curr_node = root

        # Selection phase
        while curr_node.children:
            curr_node = max(curr_node.children, key=lambda child:
                            ucb(curr_node, child))
            board_copy.push(curr_node.move)

        # Expansion phase
        if not board_copy.is_game_over():
            with torch.no_grad():
                policy, value = model(board_to_tensor(board_copy))

            move_nums = torch.zeros(16).cuda()
            for move in board_copy.legal_moves:
                new_node = Node()
                new_node.move = move
                new_node.parent = curr_node
                curr_node.children.append(new_node)
                move_nums[move[0] * 4 + move[1]] = 1
            policy *= move_nums
            curr_node.probs = (policy / torch.sum(policy)).tolist()

            value = value.item()
        else:
            # To be inverted to the correct value in backpropagation
            value = .5 if board_copy.winner == '1/2-1/2' else 0

        # Backpropagation phase
        while curr_node is not None:
            # Value is inverted to pertain to previous player
            value = 1 - value
            curr_node.wins += value
            curr_node.visits += 1
            curr_node = curr_node.parent

    return root
