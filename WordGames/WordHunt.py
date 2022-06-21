# outfile descriptor 
wordsToPlay = open("finalWords.txt", "w")
debugCnt = 0
sentWords = []

def findWords(currWord, lenCurrWord, visited, currIndex):
    
    global sentWords
    if (lenCurrWord > 6):
        return
    
    # if invalid index
    if ( (currIndex[0] < 0) or (currIndex[0] > 3) or (currIndex[1] < 0) or (currIndex[1] > 3)):
        return

    # recurse - add below
    if (currIndex[0]+1 <= 3) and [currIndex[0]+1, currIndex[1]] not in visited:
        findWords(currWord + (gameBoard[currIndex[0]+1][currIndex[1]]), lenCurrWord+1, visited + [[currIndex[0]+1, currIndex[1]]], [currIndex[0]+1, currIndex[1]])
    # recurse - add above
    if (currIndex[0]-1 >= 0) and [currIndex[0]-1, currIndex[1]] not in visited:
        findWords(currWord + (gameBoard[currIndex[0]-1][currIndex[1]]), lenCurrWord+1, visited + [[currIndex[0]-1, currIndex[1]]], [currIndex[0]-1, currIndex[1]])
    # recurse - add left
    if (currIndex[1]-1 >= 0) and [currIndex[0], currIndex[1]-1] not in visited:
        findWords(currWord + (gameBoard[currIndex[0]][currIndex[1]-1]), lenCurrWord+1, visited + [[currIndex[0], currIndex[1]-1]], [currIndex[0], currIndex[1]-1])
    # recurse - add right
    if (currIndex[1]+1 <= 3) and [currIndex[0], currIndex[1]+1] not in visited:
        findWords(currWord + (gameBoard[currIndex[0]][currIndex[1]+1]), lenCurrWord+1, visited + [[currIndex[0], currIndex[1]+1]], [currIndex[0], currIndex[1]+1])
    # recurse - add UL diag
    if ((currIndex[0]-1 >= 0) and (currIndex[1]-1 >= 0) and [currIndex[0]-1, currIndex[1]-1] not in visited):
        findWords(currWord + (gameBoard[currIndex[0]-1][currIndex[1]-1]), lenCurrWord+1, visited + [[currIndex[0]-1, currIndex[1]-1]], [currIndex[0]-1, currIndex[1]-1])
    # recurse - add UR diag
    if ((currIndex[0]-1 >= 0) and (currIndex[1]+1 <= 3) and [currIndex[0]-1, currIndex[1]+1] not in visited):
        findWords(currWord + (gameBoard[currIndex[0]-1][currIndex[1]+1]), lenCurrWord+1, visited + [[currIndex[0]-1, currIndex[1]+1]], [currIndex[0]-1, currIndex[1]+1])
    # recurse - add LL diag
    if ((currIndex[0]+1 <= 3) and (currIndex[1]-1 >= 0) and [currIndex[0]+1, currIndex[1]-1] not in visited):
        findWords(currWord + (gameBoard[currIndex[0]+1][currIndex[1]-1]), lenCurrWord+1, visited + [[currIndex[0]+1, currIndex[1]-1]], [currIndex[0]+1, currIndex[1]-1])
    # recurse - add LR diag
    if ((currIndex[0]+1 <= 3) and (currIndex[1]+1 <= 3) and [currIndex[0]+1, currIndex[1]+1] not in visited):
        findWords(currWord + (gameBoard[currIndex[0]+1][currIndex[1]+1]), lenCurrWord+1, visited + [[currIndex[0]+1, currIndex[1]+1]], [currIndex[0]+1, currIndex[1]+1])

    # check current word
    if lenCurrWord > 2 and lenCurrWord > 4:
        key = currWord[0:2]
        if key in allWords and currWord in allWords[key] and currWord not in sentWords:
            wordsToPlay.write(currWord)
            wordsToPlay.write('     ')
            wordsToPlay.write(str(visited[0]))
            wordsToPlay.write('\n')

# gather dictionary of words
allWords = {}
with open('./words.txt') as f:
    content = f.readlines()
    cntr = 1
    for entry in content:
        
        if cntr <= 2:
            cntr+=1
            continue
        elif len(entry.strip()) >= 3 and len(entry.strip()) <= 16: 
            entryClean = entry.strip()
            key = entryClean[0:2]
            if key not in allWords:
                allWords[key] = []
            allWords[key].append(entryClean)

# create structure
letterList = input('Input letters: ')
#letterList = 'ROVSDNMIHKNAHATC'
rows, cols = (4, 4)
gameBoard = [[0 for i in range(cols)] for j in range(rows)]
lettCntr = 0
for i in range(0, 4):
    for j in range(0, 4):
        gameBoard[i][j] = letterList[lettCntr]
        lettCntr+=1


# iterate through board

currWord = ''
lenCurrWord = 0

for i in range(0, 4):
    for j in range(0, 4):
        currIndex = [i, j]
        visited = [[i, j]]
        currWord = ''
        lenCurrWord = 1
        findWords(gameBoard[i][j], lenCurrWord, visited, currIndex)
        
wordsToPlay.close()

        

