#!/usr/bin/env python3

import concurrent.futures
import os
import requests
import sys
import time

# Functions

def usage(status=0):
    progname = os.path.basename(sys.argv[0])
    print(f'''Usage: {progname} [-h HAMMERS -t THROWS] URL
    -h  HAMMERS     Number of hammers to utilize (1)
    -t  THROWS      Number of throws per hammer  (1)
    -v              Display verbose output
    ''')
    sys.exit(status)

def hammer(url, throws, verbose, hid):
    ''' Hammer specified url by making multiple throws (ie. HTTP requests).

    - url:      URL to request
    - throws:   How many times to make the request
    - verbose:  Whether or not to display the text of the response
    - hid:      Unique hammer identifier

    Return the average elapsed time of all the throws.
    '''
    total = 0;
    for j in range(int(throws)):
        t1 = time.time()
        resp = requests.get(url)
        if verbose:
            print(resp.text)
        t = time.time() - t1
        print(f'Hammer: {hid}, Throw:  {j}, Elapsed Time: {round(t,2)}')
        total = total + t
    av = float(total) / float(throws);
    print(f'Hammer: {hid}, AVERAGE  , Elapsed Time: {round(av,2)}')


    return av

def do_hammer(args):
    ''' Use args tuple to call `hammer` '''
    return hammer(*args)
    

def main():
    hammers = 1
    throws  = 1
    verbose = False
    url = ''

    if len(sys.argv) == 1:
        usage(1)

    # Parse command line arguments
    for i in range(len(sys.argv)):
        if i == 0:
            continue
        if sys.argv[i-1] == '-h':
            continue
        elif sys.argv[i-1] == '-t':
            continue
        if sys.argv[i] == '-h':
            hammers = sys.argv[i+1]
        elif sys.argv[i] == '-t':
            throws = sys.argv[i+1]
        elif sys.argv[i] == '-v':
            verbose = True
        elif i == len(sys.argv)-1:
            url = sys.argv[i]
        else:
            usage(1)

            

    
    # Create pool of workers and perform throws

    arguments = ( (url, throws, verbose, i)    for i in range(int(hammers) ))

    with concurrent.futures.ProcessPoolExecutor(int(hammers)) as executor:
        totals = [float(i) for i in executor.map(do_hammer, arguments  ) ]

    av = sum(totals, 0) / float(hammers)
    print(f'TOTAL AVERAGE ELAPSED TIME: {round(av,2)}')

# Main execution

if __name__ == '__main__':
    main()

# vim: set sts=4 sw=4 ts=8 expandtab ft=python:
