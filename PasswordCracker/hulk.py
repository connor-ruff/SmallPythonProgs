#!/usr/bin/env python3

import concurrent.futures
import hashlib
import os
import string
import sys

# Constants

ALPHABET = string.ascii_lowercase + string.digits

# Functions

def usage(exit_code=0):
    progname = os.path.basename(sys.argv[0])
    print(f'''Usage: {progname} [-a ALPHABET -c CORES -l LENGTH -p PATH -s HASHES]
    -a ALPHABET Alphabet to use in permutations
    -c CORES    CPU Cores to use
    -l LENGTH   Length of permutations
    -p PREFIX   Prefix for all permutations
    -s HASHES   Path of hashes file''')
    sys.exit(exit_code)

def md5sum(s):
    ''' Compute md5 digest for given string. '''
    # TODO: Use the hashlib library to produce the md5 hex digest of the given
    # string.
    m = hashlib.md5()
    m.update(s.encode())
    return m.hexdigest()

def permutations(length, alphabet=ALPHABET):
    ''' Recursively yield all permutations of the given length using the
    provided alphabet. '''
    # TODO: Use yield to create a generator function that recursively produces
    # all the permutations of the given length using the provided alphabet.
    for a in alphabet:
        if length == 1:
            yield a
        else:
            for b in permutations(length-1, alphabet):
                yield a + str(b)

def flatten(sequence):
    ''' Flatten sequence of iterators. '''
    # TODO: Iterate through sequence and yield from each iterator in sequence.

    for a in sequence:
        for b in a:
            yield(b)
            
def crack(hashes, length, alphabet=ALPHABET, prefix=''):
    ''' Return all password permutations of specified length that are in hashes
    by sequentially trying all permutations. '''
    # TODO: Return list comprehension that iterates over a sequence of
    # candidate permutations and checks if the md5sum of each candidate is in
    # hashes.

    return [prefix+perm for perm in permutations(length,alphabet) if md5sum(prefix+perm) in hashes ]

def cracker(arguments):
    ''' Call the crack function with the specified arguments '''
    return crack(*arguments)

def smash(hashes, length, alphabet=ALPHABET, prefix='', cores=1):
    ''' Return all password permutations of specified length that are in hashes
    by concurrently subsets of permutations concurrently.
    '''
    # TODO: Create generator expression with arguments to pass to cracker and
    # then use ProcessPoolExecutor to apply cracker to all items in expression.
    arguments = ((hashes,length-1,alphabet,prefix+str(a)) for a in alphabet)

    with concurrent.futures.ProcessPoolExecutor(int(cores)) as executor:
        listy = flatten(executor.map(cracker, arguments))
    

    return listy

def main():
    arguments   = sys.argv[1:]
    alphabet    = ALPHABET
    cores       = 1
    hashes_path = 'hashes.txt'
    length      = 1
    prefix      = ''
    flags       = ['-a', '-c', '-l', '-p', '-s']

    # TODO: Parse command line arguments
    for i in range(len(arguments)):
        if arguments[i-1] in flags:
            continue
        else:
            if arguments[i] == '-h':
                usage(0)
            elif arguments[i] == '-a':
                alphabet = arguments[i+1]
            elif arguments[i] == '-c':
                cores = arguments[i+1]
            elif arguments[i] == '-l':
                length = int(arguments[i+1])
            elif arguments[i] == '-p':
                prefix = arguments[i+1]
            elif arguments[i] == '-s':
                hashes_path = arguments[i+1]

    # TODO: Load hashes set
    hashes = set(map(lambda x: x.rstrip(), open(hashes_path))) 
    
    # TODO: Execute crack or smash function
    if cores == 1 or length == 1:
        outcome = crack(hashes,length,alphabet,prefix)
    else:
        outcome = smash(hashes, length, alphabet, prefix,cores)

    # TODO: Print all found passwords
    for outie in outcome:
        print(outie)

# Main Execution

if __name__ == '__main__':
    main()

# vim: set sts=4 sw=4 ts=8 expandtab ft=python:
