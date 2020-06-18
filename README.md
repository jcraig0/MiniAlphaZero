# MiniAlphaZero

This Python program plays board games with a convolutional neural network and Monte Carlo tree search. It uses relatively few lines of code to make it easier to understand. More games may be added to the project in the future.

## Usage

Running the program requires Python 3, PyTorch, and a CUDA-enabled GPU. My environment uses Python 3.8.3, PyTorch 1.5.0, and a GeForce RTX 2060.

The command `python run.py [--game arg1]` trains a new model to play the game `arg1`, using the values specified in `config.py`. `arg1` must be either `tic-tac-toe-3` (3x3), `tic-tac-toe-4` (4x4), or `connect-4`. Every five full iterations, the best version of the model is stored as a file named `model-iter-[iteration number]`.

If more parameters are specified, they must be of the form `[--action arg2] [--model arg3]`. `arg2` is either `play_against`, which starts a human vs. computer match-up; or `solve`, in which a model evaluates a position given in `run.py`'s `solve()` function. `arg3` is the name of the model file used with `--action`. A high value for "MCTS_SIMS" is recommended for these modes.

A example 3x3 Tic-Tac-Toe model, trained for ten full iterations with the default parameters, can be found [here](https://drive.google.com/file/d/1o4pI6LYnjfd5-yBihiyJAZvJwPQx-y_q/view?usp=sharing). It must be placed inside the project folder.

## Credits

* The search and evaluation algorithms are based on [AlphaGo Zero](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ), developed by DeepMind. Here are the major differences in my implementation:
    * In the evaluation stage, "X" is always the newer version of the model.
    * Symmetries of the board are not used for training.
    * The loss function does not include a regularization term.
    * A board's tensor representation uses 1 for "X", 0 for "O", and 0.5 for a blank space.
    * The range of the value head's output is that of sigmoid (0 to 1) instead of tanh (-1 to 1).
* The UCB algorithm and neural network architecture are mostly derived from [Surag Nair's implementation](https://github.com/suragnair/alpha-zero-general) of AlphaGo Zero.
* Most method prototypes and class variable names in the board classes come from the [python-chess library](https://github.com/niklasf/python-chess) by Niklas Fiekas.
