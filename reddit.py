#!/usr/bin/env python3

import os
import sys

import requests
import pprint

# Constants

ISGD_URL = 'http://is.gd/create.php'

# Functions

def usage(status=0):
    ''' Display usage information and exit with specified status '''
    print('''Usage: {} [options] URL_OR_SUBREDDIT

    -s          Shorten URLs using (default: False)
    -n LIMIT    Number of articles to display (default: 10)
    -o ORDERBY  Field to sort articles by (default: score)
    -t TITLELEN Truncate title to specified length (default: 60)
    '''.format(os.path.basename(sys.argv[0])))
    sys.exit(status)
    

def load_reddit_data(url):
    ''' Load reddit data from specified URL into dictionary

    >>> len(load_reddit_data('https://reddit.com/r/nba/.json'))
    27

    >>> load_reddit_data('linux')[0]['data']['subreddit']
    'linux'
    '''
    # TODO: Verify url parameter (if it starts with http, then use it,
    # otherwise assume it is just a subreddit).
    if 'http' not in url:
        url = 'https://www.reddit.com/r/' + url + '/.json'
    elif '.json' not in url: 
        url = url + ".json"

    # Header That Will Allow Access to Reddit Json Data
    headers = {'user-agent': 'reddit-{}'.format(os.environ.get('USER','cse-20289-sp20'))}
    res = requests.get(url,headers=headers)
    data = res.json()
    return data['data']['children']

def shorten_url(url):
    ''' Shorten URL using is.gd service

    >>> shorten_url('https://reddit.com/r/aoe2')
    'https://is.gd/dL5bBZ'

    >>> shorten_url('https://cse.nd.edu')
    'https://is.gd/3gwUc8'
    '''
    newURL = requests.get(ISGD_URL, params={'format': 'json', 'url': url})
    newURL = newURL.json()
    return newURL['shorturl']

def print_reddit_data(data, limit=10, orderby='score', titlelen=60, shorten=False):
    ''' Dump reddit data based on specified attributes '''
    rev = False
    if orderby == 'score':
        rev = True

    posts = data
    posts = sorted(posts,key=lambda a: a['data'][orderby], reverse=rev)


    for index, post in enumerate(posts,1):
        row = ''
        if index > 1:
            print('')

        posty = post['data']['title'][:int(titlelen)]
        row = row + f"{index:>4}.\t{posty}"
        row = row + f" (Score: {post['data']['score']})"
        print(row)

        linky = post['data']['url']
        if shorten:
            linky = shorten_url(linky)
        print(f'\t{linky:>4}')
        if index == int(limit):
            break

    

def main():
    arguments = sys.argv[1:]
    url       = None
    limit     = 10
    orderby   = 'score'
    titlelen  = 60
    shorten   = False

    numArgs = int(len(sys.argv))
    if numArgs == 1 or sys.argv[1] == '-h':
        usage()

    for i in range(len(arguments)-1):
        if arguments[i] == '-s':
            shorten = True
        if arguments[i] == '-n':
            limit = arguments[i+1]
        if arguments[i] == '-o':
            orderby = arguments[i+1]
        if arguments[i] == '-t':
            titlelen = arguments[i+1]
    url = arguments.pop()

    

    data = load_reddit_data(url)
    print_reddit_data(data, limit, orderby, titlelen, shorten)
    

# Main Execution

if __name__ == '__main__':
    main()

# vim: set sts=4 sw=4 ts=8 expandtab ft=python:
