#!/usr/bin/env python3

import os
import sys

# Functions

def usage(status=0):
    ''' Display usage message and exit with status. '''
    progname = os.path.basename(sys.argv[0])
    print(f'''Usage: {progname} [flags]

    -c      Prefix lines by the number of occurences
    -d      Only print duplicate lines
    -i      Ignore differences in case when comparing, prints out full line in lowercase
    -u      Only print unique lines

By default, {progname} prints one of each type of line.''')
    sys.exit(status)

def count_frequencies(stream=sys.stdin, ignore_case=False):
    ''' Count the line frequencies from the data in the specified stream while
    ignoring case if specified. '''
   
    dicty = {} 
    for line in stream:
        if ignore_case:
            line = line.lower()
        dicty[line.rstrip()] = (dicty.get(line.rstrip(),0)+1)

    return dicty


    
            
   

    

def print_lines(frequencies, occurrences=False, duplicates=False, unique_only=False):
    ''' Display line information based on specified parameters:

    - occurrences:  if True, then prefix lines with number of occurrences
    - duplicates:   if True, then only print duplicate lines
    - unique_only:  if True, then only print unique lines
    '''
    
    for key in frequencies:
        if not duplicates and not unique_only:
            if occurrences:
                print(f'{frequencies[key]:>7} {key}')
            else:
                print(key)
        elif duplicates and frequencies[key] > 1: 
            if occurrences:
                print(f'{frequencies[key]:>7} {key}')
            else:
                print(key)
        elif unique_only and frequencies[key] == 1:
            if occurrences:
                print(f'{frequencies[key]:>7} {key}')
            else:
                print(key)
            

def main():
    ''' Process command line arguments, count frequencies from standard input,
    and then print lines. '''
    dicty = { }
    arguments = sys.argv[1:]
    prefix = False
    dupes = False
    ignorecase = False
    onlyunique = False

    while arguments and arguments[0].startswith('-'):
        arg = arguments.pop(0)

        if arg == '-i':
            ignorecase = True
        if arg == '-c':
            prefix = True
        if arg == '-d':
            dupes = True
        if arg == '-u':
            onlyunique = True
        if arg == '-h':
            usage(0)

    dicty = count_frequencies(sys.stdin, ignorecase)
    print_lines(dicty, prefix, dupes, onlyunique)
    

# Main Execution

if __name__ == '__main__':
    main()

# vim: set sts=4 sw=4 ts=8 expandtab ft=python:
