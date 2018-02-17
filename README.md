# Deep Neural Decoders for Fault-Tolerant Quantum Error Correction

This repository contains source codes and results reported in 

C. Chamberland, P. Ronagh, 
*Deep neural decoders for near term fault-tolerant experiments*, arXiv 2018

The package provides a general framework for training and using deep learning
for fault tolerant protocols that use CSS codes. The methods implemented here
can have practical applications for near term fault-tolerant experiments.

### Prerequisites

The implementation has been tested with Mac OS 10.12, Mac OS 10.13, and on Linux
Debian 9 (4.9.65-3+deb9u1 (2017-12-23) x86_64 and 4.9.51-1 (2017-09-28) x86_64)
but should work with most Mac OSX and Linux systems. To run the codes you need
to have the following installed.

* Python 2.7 
* [BayesOpt](https://github.com/rmcantin/bayesopt) - A Bayesian optimization 
  library
* [Tensorflow](https://www.tensorflow.org/) Tensorflow 1.4

To run the circuit simulations Matlab is required.

### Reports

The following neural networks were tested and the results are available in the
[Reports](Reports) folder. The number of hidden layers for feedforward networks
was incremented until no further improvement was observed in the performance.

| FTEC    | T  | D | RNN | FF0 | FF1 | FF2 | FF3 | FF4 | CNN |  B Range  | Tune |
| ------- | -- | - | --- | --- | --- | --- | --- | --- | --- | --------- | ---- |
| Steane  | PU | 3 | ### |     | ### | ### | ### |     |     | 1e-4 6e-4 | 4e-4 |
| Steane  | LU | 3 | ### | ### | ### | ### | ### | ### |     | 1e-4 6e-4 | 4e-4 |
| Knill   | PU | 3 | ### |     |     | ### | ### |     |     | 1e-4 6e-4 | 4e-4 |
| Knill   | LU | 3 | ### | ### | ### | ### | ### | ### |     | 1e-4 6e-4 | 4e-4 |
| Surface | PU | 3 | ### | ### | ### | ### |     |     |     | 1e-4 6e-4 | 4e-4 |
| Surface | LU | 3 | ### | ### | ### | ### |     |     |     | 1e-4 6e-4 | 4e-4 |
| Steane  | PU | 5 | ### |     |     | ### | ### |     |     | 6e-4 2e-3 | 1e-3 |
| Steane  | LU | 5 | ### |     |     | ### | ### |     |     | 6e-4 2e-3 | 1e-3 |
| Knill   | PU | 5 |     |     |     | ### |     |     |     | 6e-4 2e-3 | 4e-4 |
| Knill   | LU | 5 |     |     |     | ### |     |     |     | 6e-4 2e-3 | 4e-4 |
| Surface | PU | 5 | ### |     |     | ### |     |     | ### | 3e-4 8e-4 | 6e-4 |
| Surface | LU | 5 | ### |     |     | ### |     |     | ### | 3e-4 8e-4 | 6e-4 |

* *PU*. Pure-Error quantum error correction as base decoder
* *LU*. Lookup table qunatum error correction as base decoder
* *Steane*. Steane EC protocol for fault-tolerant error correction. In this
experiment the exRec CNOT was simulated.
* *Knill*. Knill EC protocol for fault-tolerant error correction. In this
experiment the exRec CNOT was simulated.
* *Surface*. Rotated surface code single EC in fault-tolerant protocols (3 
rounds of EC for surface code of distance 3, and 6 rounds of EC for distance 5).
* *B Range*. Range of physical error rates used for depolarizing noise channel.
* *Tune*. Physical error rate at which hyperparameter tuning was performed
using BayesOpt.

### Reproducing the results

Here I explain how to reproduce our results and/or build upon them. Feel free to
[contact me](https://pooya-git.github.io/) for further details, discussion,
issues and feedback. 

#### Pickle your dataset

The training mode uses pickle files. To generate pickles from your dataset use

```
python -u Run.py gen param_file
```

where `param_file` is the addresss of a `json` file like
[this](Param/LookUp/Steane_CNOT_D5/ff_init_param/1.json).

Containing the following keys in this example:
```
    "FT scheme": "ExRecCNOT", 
    "EC scheme": "ColorD5", 
    "raw folder": "../../Data/Compact/Steane_CNOT_D5/",
    "pickle folder": "../../Data/Pkl/LookUp/Steane_CNOT_D5/",
    "report folder": "../../Reports/LookUp/Steane_CNOT_D5/",
    "param folder": "../../Param/LookUp/Steane_CNOT_D5/",
    "look up": true
```

`raw folder` is a text file with zeros and ones in the format explained below.
The results will be saved in `pickle folder`. If the training set is for the
PE-based method the `look up` should be set to `false`.

#### Hyperparameter tuning

The choice of neural network to use and the hyperparameters used by it can be
set in a `json` file like 
[this](Param/LookUp/Steane_CNOT_D5/2018-01-15-10-40-15.json).

You can alternatively use hyperparameter tuning feature I implemented:

```
python -u Run.py tune param_file hyperparam_range_file
```

`param_file` is again a param like, e.g. 
[this](Param/LookUp/Steane_CNOT_D5/ff_init_param/1.json).
`hyperparam_range_file` is a file putting box-constrains on the range of
each hyperparam in the domain explored in Bayesian optimization, E.g.
[this](Param/LookUp/Steane_CNOT_D5/hyperparam.json).

#### Train using your pickle

Once you have set all your parameters in a `json` file like
[this](Param/LookUp/Steane_CNOT_D5/2018-01-15-10-40-15.json)
then run

```
python -u Run.py bench training_param_file [index_0] [index_1]
```
`index_0` and `index_1` lets you do training and benchmarking on a small range 
of files available in `pickle folder`.

#### Add your favorite neural network

There are many neural networks implemented in TensorFlow in 
[Networks.py](Trainer/Networks.py). Many of them proved useless. I will let them
be there as examples of what I have tried. In the paper, only the following
neural networks are used:

* surface_conv3d_cost
* ff_cost
* rnn_cost

You can add your model and a line to `cost_function` in `Model.py` and you are
good to go.

#### Add your favorite CSS code

Examples `_*.py` shows how to add a new CSS code.
Choose a name and use it in `env` parameter of the `param` file. The new
`_*.py` file has to be imported as `lookup` like in 
[this line](https://github.com/pooya-git/DeepNeuralDecoder/blob/b275aed0192c4860b9da2de20e0c2734f15a302c/Trainer/Run.py#L130).


## License

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details.

