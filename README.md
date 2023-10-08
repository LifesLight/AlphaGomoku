# Self learning Gomoku AI

## Basics:
In Gomoku black starts.
In code black is 0, white is 1.

Gomoku board representations are defined as follows:
Tensor of shape `[HistoryDepth(HD) + 1, BoardSize, BoardSize]` where HD is a even number and at least 2 (HD 2 is no history). Axis 0 is described as:

`[0]: Next players color`

`[1 - HD/2]: Black stone history`

`[HD/2+1 - HD]: White stone history`

This makes it so we can go back HD - 2 played moves on the board.
For each step back (index - 1) the last placed stone of given color is removed from that board.
A interface to the history is provided in Utilities.py.

## Datasets
The human datasets have following naming convention: `HD(HistoryDepth),[AUG (is augmented)],TS(TrainsplitFraction),Rulesets(All rulesets in the dataset)`.

## Compatibility
Tested on Pytorch version 2.1.0 under Linux