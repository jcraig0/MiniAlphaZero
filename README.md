# SimpleAlphaZero

This Python program plays 4x4 tic-tac-toe with a convolutional neural network and Monte Carlo tree search. It uses relatively few lines of code to make it easier to understand. More board games may be added to the project in the future.

## Usage

Running the program requires Python 3, PyTorch, and a CUDA-enabled GPU. My environment uses Python 3.8.3, PyTorch 1.5.0, and a GeForce RTX 2060.

Executing `run.py` without any parameters trains a new model to play tic-tac-toe, using the values specified in `config.py`. Every five full iterations, the best version of the model is stored as a file.

If parameters are specified, they must be of the form `[--action arg1] [--model arg2]`. `arg1` is either `play_against`, which starts a human vs. computer match-up with the human as "X"; or `solve`, in which a model evaluates a position given in `run.py`'s `solve()` function. `arg2` is the name of the model file used with `--action`.

## Credits

* The search and evaluation algorithms are based on [AlphaGo Zero](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ), developed by DeepMind. Here are the major differences in my implementation:
    * In the evaluation stage, "X" is always the newer version of the model.
    * Symmetries of the board are not used for training.
    * The loss function does not include a regularization term.
    * The range of the value head's output is that of sigmoid (0 to 1) instead of tanh (-1 to 1).
* The UCB algorithm and neural network architecture are mostly based on [Surag Nair's implementation](https://github.com/suragnair/alpha-zero-general) of AlphaGo Zero.
* Most method prototypes and class variable names in `tic_tac_toe.py` come from the [python-chess library](https://github.com/niklasf/python-chess) by Niklas Fiekas.