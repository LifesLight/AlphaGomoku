# Self learning Gomoku AI

## Patchnotes
**v.0.1.0 -> v.0.1.1**<br>
Added graphviz output for trees!

TODO: 
- Rewrite Python codebase: Goal selfplay / selftrain loop
- Implement RIF ruleset in both Python and C++

## Basics
**In Gomoku black starts.** In code:<br>
Black is 0 and -1.0 is black winning evaluation<br>
White is 1 and 1.0 is white winning evaluation<br>
(Draw is 2 and 0.0 is drawn evaluation)<br>

## Algorithm
The general idea is to use a variation of MCTS, which uses a Neural Network with policy and value-head.<br>
The Neural Networks parameters are improved over time via selfplay. The model is retrained on better moves calculated by the MCTS.
If the new model wins at least 55% of games against the old one keep it.<br>

*This loop is not yet implemented*

## Multithreading
Multithreading is implemented via a worker pool.<br>
When creating a batcher it calculates the amount of **GCP** (Gamestate conversion processes), and **SIM** (Simulation) workers based on provided [hyperparameters](#threadingParam).
These workers are all the workers ever created by that Batcher, we dynamically reduce the amount of workers to start when needed but we never create new ones.<br>
Multithreading is implemented on a Batcher level since every environment is independent of all other, which makes multithreading fairly efficient due to not needing any mutexes or atomics. (Every environment will only ever be worked on by one thread).<br>

### <a name="threadingParam"></a> Hyperparameters for threading are:
- PerThreadSimulations: How many threads for simulating (MCTS Tree).
- PerThreadGamestateConvertions: How many threads for converting nodes to [gamestate](#gs) tensors.

*For n environments*

- MaxThreads: How many max threads per type (can result in twice the amount of workers then expected but different types will never be active simultaniously).

## Data
### <a name="gs"></a>Gamestate
Gomoku gamestates for neural network input are encoded as follows:<br>
Tensor of shape **[HistoryDepth(HD) + 1, BoardSize, BoardSize]**.<br>**HD is a even number and at least 2** *(HD 2 is no history)*.<br>Axis 0 is described as:<br><br>
**[0]:** Next players color<br>
**[1 - HD/2]:** Black stone history<br>
**[HD/2+1 - HD]:** White stone history<br><br>
This makes it so we encode the last **HD - 2** played moves on the board.<br>
For each step back (index - 1) the last placed stone of given color is removed from that board.<br>
#### Reading/Writing
**Python:**<br>
Gamestates can be sliced (render history board states) via the **sliceGamestate** function in **Utilities.py**<br>
Datapoints are only ever written by the **DatasetCreator.py**<br>
**C++:**<br>
The **sliceGamestate** function also exists in **Utilities.h**<br>
Gamestates are created from Nodes via the **nodeToGamestate** function in **Node.cpp**<br>
*(Reading gamestates should never be necessary except for debugging)*

### Datapoint
A datapoint contains:<br>
- History of moves that lead to gamestate
- The best move at gamestate according to MCTS
- If current player won
### Selfplay
In selfplay the last 500.000 datapoints are stored so we can sample datapoints for retraining.<br>
They are stored in a text file, reading and writing is only recommended via the **Storage** class.

### Naming Conventions
**Datasets from human data:**<br> HD(HistoryDepth),[AUG (is augmented)],TS(TrainsplitFraction),Rulesets(All rulesets in the dataset)<br>

## Neural Network
### Architecture
Filter and Layer count are by user choice<br><br>
[**Gamestate**](#gs)<br>
&nbsp; &nbsp; &nbsp; &nbsp; ↓<br>
&nbsp; &nbsp;[**ResNet**](#resnet) ↺<br>
&nbsp; &nbsp; &nbsp; &nbsp; ↓ &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp;↓<br>
[**Policy Head**](#polhead) &nbsp;&nbsp; [**Value Head**](#valhead)


### <a name="resnet"></a>ResNet
<pre>
x = input
residual = x
    Convolution: 3x3 Kernal with padding
    Batchnorm
    ReLU
    Convolution: 3x3 Kernal with padding
    Batchnorm
x + residual
ReLU
</pre>

### <a name="polhead"></a>Policy Head
<pre>
Convolution: 1x1 Kernal with 2 Filters
Flatten
Batchnorm
ReLU
Linear: 450 → 225
(Softmax)
</pre>

### <a name="valhead"></a>Value Head
<pre>
7 times:
    Convolution: 3x3 without padding
    Batchnorm
    ReLU
Flatten
Linear: Filters → Linear Filters
Batchnorm
ReLU
Linear: Linear Filters → Linear Filters
Batchnorm
ReLU
Linear: Linear Filters → 1
Tanh
</pre>

## Rules
*Proper rules not yet implemented.*<br>

For now just "freestyle" 5 in a row wins.<br>

## <a name="modes"></a>Modes
The AlphaGomoku executable can be called with 1 of 3 modes:<br>
- **DUEL:** Evaluate 2 models against each other (used in retrain validation).<br>
- **SELFPLAY:** Let model play against itself to generate datapoints for retraining.<br>
- **HUMAN:** Lets you play against a model with MCTS.<br>

## Environment Variables
**LOGGING:**
- INFO: Logs non verbose information
- WARNING: Logs warnings, usually not terminal
- ERROR: Logs errors, can be terminal
- FATAL: Logs terminal errors

## Parameters
- help                    : Print help message.
- [mode](#rules)          : Mode to run the program in (duel, selfplay, human).
- *model*                 : Name of the model.
- *simulations*           : Number of simulations to run per move.
- environments            : Number of environments to run in parallel.
- randmoves               : Number of random moves to make before starting.
- humancolor              : Color of the human player (0 = black, 1 = white).
- stones                  : Stone skin to use for rendering.
- board                   : Board skin to use for rendering.
- renderenvs              : Render the environments.
- renderanalytics         : Render the analytics.
- renderenvscount         : Number of environments to render.
- datapath                : Path to store the data.
- modelpath               : Path to store the models.
- *device*                : Device to use for inference (cpu, cuda, mps).
- *scalar*                : Scalar to use for inference (float16, float32).
- threads                 : Number of threads to use for batching.
- batchsize               : Batchsize cap for inference.
- policybias              : Policy bias to use for MCTS.
- valuebias               : Value bias to use for MCTS.
- explorationbias         : Exploration bias to use for MCTS.
- modelpath               : Where models are stored.
- datapath                : Where datapoints are stored.
- outputrees              : Weather trees should be output as graphviz files
- outputtreespath         : Where graphviz tree outputs are stored

*Italic* args can pe specified per model like: --device1 [model1 device] --device2 [model2 device].

## Compatibility
**Tested with Pytorch 2.1.1 on:**<br>
Ubuntu 22.04 - (CUDA.12.1)<br>
MacOS 17.1 - (MPS)