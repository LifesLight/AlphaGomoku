# Self learning Gomoku AI

**Basics:**
In Gomoku black starts.
Black is represented as 1, white as 0.

Gomoku board representations are defined as follows:
Tensor of shape [HistoryDepth(HD) * 2 + 1, BoardSize, BoardSize], where axis 0 is:
`[0]:`          Next players color
`[1 - HD]:`     White stone history
`[HD+1 - HD*2]:`Black stone history

Each history entry stores all stones of said color on the board at given time.
The larger the indecies the older the position.

Ideas:
Make history a "history per color move" so that we don't store the same data twice for each move.