import xml.etree.ElementTree as ET
import numpy as np
import os

# -----Generation Parameters------ #

DATASET_SOURCE = 'RenjunetDatasets/renjunet_v10.xml'
HD = 7
AUGMENTED = False
# ['all'] for all
RULESETS_WHITELIST = ['all']
RULESETS_BLACKLIST = [7]
TRAINSPLIT = 0.8
TARGETFOLDER = 'GeneratedDatasets'

# -------------------------------- #


class DatasetGame:
    def __init__(self):
        self.ID = None
        self.Opening = None
        self.Ruleset = None
        self.Players = {"Black" : None, "White" : None}
        self.Winner = None
        self.Tourney = None
        self.Moves = []
        self.Augmented = False

class GameHandler:
    def __init__(self, Game):
        self.moves = Game.Moves
        self.index = 0

    def onLastMove(self):
        return self.index >= len(self.moves) - 1
    
    def getMoveHistory(self):
        moveHistory = []
        for iterator, move in enumerate(self.moves[:self.index]):
            moveHistory.append((move, iterator % 2))
        return moveHistory

    def getNLastMoves(self, n):
        lastMoves = []
        for iteratorIndex in range(self.index - n - 1, self.index - 1):
            if iteratorIndex < 0:
                lastMoves.append((None, iteratorIndex % 2))
            else:
                lastMoves.append((self.moves[iteratorIndex], iteratorIndex % 2))
        return lastMoves
            
    def nextMove(self):
        self.index += 1
        if self.index >= len(self.moves):
            return None
        return (self.moves[self.index], self.index % 2)

def formatRanges(numbers):
    ranges = []
    start = end = numbers[0]
    for num in numbers[1:]:
        if num == end + 1:
            end = num
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = num
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ", ".join(ranges)

def MovesToGamestate(moves):
    numpyGamestate = np.zeros((HD * 2 + 1, 15, 15), dtype=bool)

    toMoveColor = (moves[-1][1] + 1) % 2
    numpyGamestate[0, :, :] = toMoveColor
    
    generalMoves = moves[:len(moves) - HD]
    slicedMoves = moves[len(moves) - HD:]

    for move, color in generalMoves:
        if color:
            for i in range(1, HD + 1):
                numpyGamestate[i][move] = True
        else:
            for i in range(HD + 1, HD * 2 + 1):
                numpyGamestate[i][move] = True

    for iterator, (move, color) in enumerate(slicedMoves):
        for i in range(iterator, -1, -1):
            if color:
                numpyGamestate[i + 1][move] = True
            else:
                numpyGamestate[i + HD + 1][move] = True

    return numpyGamestate

def gamestateFromBytes(gamestateBytes):
    return np.frombuffer(gamestateBytes, dtype=bool).reshape((HD * 2 + 1, 15, 15))

def MoveToString(move):
    move = move[0]
    return f'{move[0]},{move[1]}'

def StringToMove(s):
    x, y = map(int, s.split(','))
    return x, y

def MoveToGamestateTarget(move):
    target = np.zeros((15, 15), dtype=bool)
    target[move] = True
    target = target.flatten()
    return target


RULESETS = []
if RULESETS_WHITELIST[0] == 'all':
    RULESETS = [i for i in range(1, 30)]
else:
    RULESETS = RULESETS_WHITELIST[:]
for blacklisted in RULESETS_BLACKLIST:
    if blacklisted in RULESETS:
        RULESETS.remove(blacklisted)
if len(RULESETS) == 0:
    raise Exception("No rulesets selected")

RULESETS_STR = formatRanges(sorted(RULESETS))
PATH = f'{TARGETFOLDER}/HD{HD},{"AUG," if AUGMENTED else ""}TS{TRAINSPLIT},RULESETS({RULESETS_STR})'

print(f'Generating Dataset with HD={HD}, Augmented={AUGMENTED}, TS={TRAINSPLIT}, Rulesets={RULESETS_STR}')

datasetGames = []
xmlTree = ET.parse(DATASET_SOURCE)
xmlRoot = xmlTree.getroot()
xmlGames = xmlRoot.find("./games")


for game in xmlGames:
    datasetGame = DatasetGame()
    datasetGame.ID = game.attrib.get("id", None)
    datasetGame.Opening = game.attrib.get("opening", None)
    datasetGame.Tourney = game.attrib.get("tournament", None)
    datasetGame.Ruleset = game.attrib.get("rule", None)
    datasetGame.Winner = game.attrib.get("bresult", None)
    datasetGame.Players["Black"] = game.attrib.get("black", None)
    datasetGame.Players["White"]  = game.attrib.get("white", None)
    
    xmlMoves = game.find("./move").text
    if xmlMoves is not None:
        moveOrder = []
        for move in xmlMoves.split(" "):
            moveOrder.append((ord(move[0]) - 97, int(move[1:]) - 1))
        datasetGame.Moves = moveOrder
        datasetGames.append(datasetGame)
del xmlTree, xmlRoot, xmlGames

print(f'Extracted {len(datasetGames)} Games from: {DATASET_SOURCE}')
#for game in datasetGames:
#    print(f'ID: {game.ID}, Ruleset: {game.Ruleset}, Opening: {game.Opening}, Tourney: {game.Tourney}, White: {game.Players["White"]}, Black: {game.Players["Black"]}, Winner: {game.Winner}, Moves:')
#    print(game.Moves)
#    break

print("Generating Gamestates...")

if AUGMENTED:
    print("Augmenting Dataset...")
    augmentedDatasetGames = []
    for game in datasetGames:
        moves = game.Moves
    
        variants = [[] for i in range(5)]
        for move in moves:
            # Mirror Move X
            variants[0].append((14 - move[0], move[1]))
            # Mirror Move Y
            variants[1].append((move[0], 14 - move[1]))
            # 90 Degree Rotation
            variants[2].append((14 - move[1], move[0]))
            # 180 Degree Rotation
            variants[3].append((14 - move[0], 14 - move[1]))
            # 270 Degree Rotation
            variants[4].append((move[1], 14 - move[0]))
    
        augmentedDatasetGames.append(game)
        for i in range(5):
            newGame = DatasetGame()
            newGame.ID = game.ID
            newGame.Opening = game.Opening
            newGame.Tourney = game.Tourney
            newGame.Ruleset = game.Ruleset
            newGame.Winner = game.Winner
            newGame.Players = game.Players
            newGame.Augmented = True
            newGame.Moves = variants[i]
            augmentedDatasetGames.append(newGame)
    datasetGames = augmentedDatasetGames
    del augmentedDatasetGames
    
    print(f'Augmented Dataset to {len(datasetGames)} Games')

gamestateToNextMoves = {}
for game in datasetGames:
    if int(game.Ruleset) not in RULESETS:
        continue

    gameHandler = GameHandler(game)
    
    while not gameHandler.onLastMove():
        nextMove = gameHandler.nextMove()
        nextMoveString = MoveToString(nextMove)
        lastMoves = gameHandler.getMoveHistory()
        gamestate = MovesToGamestate(lastMoves)
        gamestateBytes = gamestate.tobytes()
        
        if gamestateBytes in gamestateToNextMoves.keys():
            gamestateToNextMoves[gamestateBytes].append(nextMoveString)
        else:
            gamestateToNextMoves[gamestateBytes] = [nextMoveString]
     
gamestateToMoveCountDict = {}
for key, values in gamestateToNextMoves.items():
    gamestateToMoveCountDict[key] = {}
    for value in values:
        if value in gamestateToMoveCountDict[key].keys():
            gamestateToMoveCountDict[key][value] += 1
        else:
            gamestateToMoveCountDict[key][value] = 1
del gamestateToNextMoves

gamestateToNextMove = {}
for key, value in gamestateToMoveCountDict.items():
    mostPlayedMove = max(value, key=lambda k: value[k])
    gamestateToNextMove[key] = mostPlayedMove


finishedDataset = []
for key, value in gamestateToNextMove.items():
    x = gamestateFromBytes(key)
    nextMove = StringToMove(value)
    y = MoveToGamestateTarget(nextMove)
    finishedDataset.append((x, y))
del gamestateToNextMove

print(f'Generated {len(finishedDataset)} gamestates')
print('Writing to disk...')

trainSizePercentile = TRAINSPLIT
outputPath = PATH
os.makedirs(outputPath, exist_ok=True)

np.random.shuffle(finishedDataset)
index = int(trainSizePercentile * len(finishedDataset))

np.array([x[0] for x in finishedDataset[:index]], dtype=bool).tofile(f'{outputPath}/XTrain.bin')
np.array([x[1] for x in finishedDataset[:index]], dtype=bool).tofile(f'{outputPath}/YTrain.bin')
np.array([x[0] for x in finishedDataset[index:]], dtype=bool).tofile(f'{outputPath}/XTest.bin')
np.array([x[1] for x in finishedDataset[index:]], dtype=bool).tofile(f'{outputPath}/YTest.bin')

print(f'Saved Dataset to {outputPath}')