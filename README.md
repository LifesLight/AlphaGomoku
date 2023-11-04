# Self learning Gomoku AI

## Basics:
In Gomoku black starts.
In code black is 0, white is 1.

Gomoku board representations are defined as follows:
Tensor of shape `[HistoryDepth(HD) + 1, BoardSize, BoardSize]` where **HD is a even number and at least 2** (HD 2 is no history). Axis 0 is described as:

`[0]:` Next players color<br>
`[1 - HD/2]:` Black stone history<br>
`[HD/2+1 - HD]:` White stone history<br>

This makes it so we can go back HD - 2 played moves on the board.
For each step back (index - 1) the last placed stone of given color is removed from that board.
A interface to the history is provided in Utilities.py.

## Datasets
The human datasets have following naming convention:<br> `HD(HistoryDepth),[AUG (is augmented)],TS(TrainsplitFraction),Rulesets(All rulesets in the dataset)`.

## Environment Variables
`LOGGING:` INFO - Logs non verbose information<br>
`RENDER_ENVS_COUNT:` How many environments should be rendered for following:<br>
`RENDER_ENVS:` Enable rendering of environments in autoplay loops (Like duel models)<br>
`RENDER_ANALYTICS:` If TRUE also renders analytic information for each environment<br>

## Compatibility
Tested on Pytorch version 2.1.0 (CUDA.12.1) under Linux