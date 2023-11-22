import torch
import subprocess
import numpy as np
from Utilities import Utilities as Utils
from Config import Config as Conf
import sys
import random as rand

DATAPOINT_PATH = "../../Datasets/Selfplay/data.txt"

class Datapoint:
    def __init__(self, datapointString):
        rawMoves = []
        # Parse the string
        i = 0
        # Parse moves
        splittedComma = datapointString.split(",")
        rawMoves = splittedComma[:-1]
        splittedSemicolon = splittedComma[-1].split(";")
        rawMoves.append(splittedSemicolon[0])
        self.bestMove = splittedSemicolon[1]
        self.result = splittedSemicolon[2]

        self.moves = []
        for i, move in enumerate(rawMoves):
            # Convert string to int
            intMove = int(move)
            x, y = Utils.indexToCords(intMove)
            self.moves.append(((x, y), i % 2))


    def getGamestate(self):
        moves = self.moves
        HD = Conf.HISTORYDEPTH
        halfHistory = HD // 2
        numpyGamestate = np.zeros((HD + 1, 15, 15), dtype=bool)
        toMoveColor = not moves[-1][1]
        numpyGamestate[0, :, :] = toMoveColor
        generalMoves = moves[:max(len(moves) - HD, 0)]
        slicedMoves = moves[max(len(moves) - HD, 0):]
        for move, color in generalMoves:
            if color:
                for i in range(halfHistory + 1, halfHistory * 2 + 1):
                    numpyGamestate[i][move] = True
            else:
                for i in range(1, halfHistory + 1):
                    numpyGamestate[i][move] = True

        for iterator, (move, color) in enumerate(slicedMoves):
            localIterator = iterator // 2
            if color:
                for i in range(localIterator + 1):
                    index = HD - i
                    numpyGamestate[index][move] = True
            else:
                for i in range(localIterator + 1):
                    index = halfHistory - i
                    numpyGamestate[index][move] = True

        return numpyGamestate


    def getResult(self):
        return self.result

    def getBestMove(self):
        return self.bestMove

class DatapointSampler:
    def __init__(self, path):
        file = open(path, 'r')
        self.datapoints = []
        while True:
            line = file.readline()
            if not line:
                break
            self.datapoints.append(line[:-1])
        rand.shuffle(self.datapoints)
        file.close()
        self.itterator = 0

    def sample(self):
        x = Datapoint(self.datapoints[self.itterator])
        self.itterator += 1
        if (self.itterator >= len(self.datapoints)):
            self.itterator = 0
            rand.shuffle(self.datapoints)
            print("Reshuffled data")
        return x

# Get commandline params
argv = sys.argv
sampler = DatapointSampler(DATAPOINT_PATH)

print(sampler.sample().getGamestate())