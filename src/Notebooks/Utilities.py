class Utilities:
    def sliceGamestate(gamestate, HD, depth):
        output = ""
        halfDepth = HD // 2
        if HD == 2:
            blackStones = gamestate[2][:][:]
            whiteStones = gamestate[1][:][:]
        else:
            tempDepth = depth // 2

            whiteIndex = halfDepth - tempDepth
            blackIndex = halfDepth * 2 - tempDepth

            if depth % 2 == 1:
                if gamestate[0][0][0]:
                    whiteIndex -= 1
                else:
                    blackIndex -= 1

            blackStones = gamestate[blackIndex][:][:]
            whiteStones = gamestate[whiteIndex][:][:]
        output += "     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14\n"
        output += "   --------------------------------------------------------------\n"
        for y in range(14, -1 ,-1):
            output += f'{y:2} |'
            for x in range(15):
                if blackStones[x][y] == 0 and whiteStones[x][y] == 0:
                    output += "   "
                elif blackStones[x][y] == 1:
                    output += " B "
                elif whiteStones[x][y] == 1:
                    output += " W "
                output += "|"
            output += "\n   --------------------------------------------------------------\n"
        return output
    
    def sliceLegacyGamestate(gamestate, HD, depth):
        output = ""
        if HD == 1:
            blackStones = gamestate[2][:][:]
            whiteStones = gamestate[1][:][:]
        else:
            blackStones = gamestate[HD * 2 - depth][:][:]
            whiteStones = gamestate[HD - depth][:][:]
        output += "     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14\n"
        output += "   --------------------------------------------------------------\n"
        for y in range(14, -1, -1):
            output += f'{y:2} |'
            for x in range(15):
                if blackStones[x][y] == 0 and whiteStones[x][y] == 0:
                    output += "   "
                elif blackStones[x][y] == 1:
                    output += " B "
                elif whiteStones[x][y] == 1:
                    output += " W "
                output += "|"
            output += "\n   --------------------------------------------------------------\n"
        return output