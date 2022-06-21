from english_words import english_words_lower_alpha_set
import random


def loadTree(allWords):
    pass

def playGame():

    # Load Words
    allWords = []
    for entry in english_words_lower_alpha_set:
        if len(entry) ==5:
            allWords.append(entry)

    candidates = []
    for entry in allWords:
        if entry[1] != 'o' or entry[3] != 's' or entry[4] != 'e':
            continue
        else:
            candidates.append(entry)

    print(candidates)
    return
        

    wordToSolve = random.choice(allWords)
    print(allWords[:10])

    # Prompt User 
    attempts = 0
    solved = False
    finalWord = ''
    while (solved == False and attempts < 5):
        attempts +=1
        usrGuess = ''
        while len(usrGuess) != 5 or usrGuess not in allWords:
            usrGuess = input(f'Attempt {attempts}:  ').lower()
        
        output = ''
        j = 0
        for letter in usrGuess:
            if letter not in wordToSolve:
                output = output + 'X'
            elif letter == wordToSolve[j]:
                output = output + '0'
            else:
                output = output + '#'
            j+=1

        if wordToSolve == usrGuess:
            solved = True
            break
            
        print(f'Output: {output}')
        

    if solved == True:
        print('Nice!')
    else:
        print('Oof! Maybe Next Time!')


playGame()



