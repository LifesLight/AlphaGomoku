# Self learning Gomoku AI (PREVIEW)

## Basics
**In Gomoku black starts.** In code:<br>
Black is 0 and -1.0 is black winning evaluation<br>
White is 1 and 1.0 is white winning evaluation<br>
(Draw is 2 and 0.0 is drawn evaluation)<br>

## Algorithm
The general idea is to use a variation of MCTS which uses a Neural Network with policy and value-head.<br>
The Neural Networks parameters are improved over time via selfplay. The model is retrained on better moves calculated by the MCTS.
If the new model wins at least 55% of games against the old one keep it.<br>

*This loop is not yet implemented*

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
&nbsp; &nbsp;[**ResNet**](#resnet)<br>
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

## Environment Variables
**LOGGING:**<br> INFO: Logs non verbose information<br>
**RENDER_ENVS:** If TRUE render environments in autoplay loops (Like duel models)<br>
**RENDER_ANALYTICS:** If TRUE render analytic information for each environment (requires ↑)<br>
**RENDER_ENVS_COUNT:** How many environments should be rendered (for ↑↑)<br>

## Compatibility
**Tested with Pytorch 2.1.0 on:**<br>
Ubuntu 22.04 - (CUDA.12.1)<br>
MacOS 17.1 - (MPS)