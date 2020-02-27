#!/usr/bin/env python3

import os
import sys

# Globals

OPERATORS = {'+', '-', '*', '/'}

# Functions

def usage(status=0):
    ''' Display usage message and exit with status. '''
    progname = os.path.basename(sys.argv[0])
    print(f'''Usage: {progname}

By default, {progname} will process expressions from standard input.''')

    sys.exit(status)

def error(message):
    ''' Display error message and exit with error. '''
    print(message, file=sys.stderr)
    sys.exit(1)

def evaluate_operation(operation, operand1, operand2):
    ''' Return the result of evaluating the operation with operand1 and
    operand2.

    >>> evaluate_operation('+', 4, 2)
    6

    >>> evaluate_operation('-', 4, 2)
    2

    >>> evaluate_operation('*', 4, 2)
    8

    >>> evaluate_operation('/', 4, 2)
    2.0
    '''
    if (operation == '+'):
        return operand1 + operand2
    elif (operation == '-'):
        return operand1 - operand2
    elif (operation == '*'):
        return operand1 * operand2
    elif (operation == '/'):
        return operand1 / operand2

    
def evaluate_expression(expression):
    ''' Return the result of evaluating the RPN expression.

    >>> evaluate_expression('4 2 +')
    6.0

    >>> evaluate_expression('4 2 -')
    2.0

    >>> evaluate_expression('4 2 *')
    8.0

    >>> evaluate_expression('4 2 /')
    2.0

    >>> evaluate_expression('4 +')
    Traceback (most recent call last):
    ...
    SystemExit: 1

    >>> evaluate_expression('a b +')
    Traceback (most recent call last):
    ...
    SystemExit: 1
    '''
    listy = [ ]
    for part in expression.split():
        if part in OPERATORS:
            try:
                part = evaluate_operation(part, listy[-2], listy[-1])
                listy.pop()
                listy.pop()
            except IndexError:
                error('Not Enough Operands')
        else:
            try:
                part = float(part)
            except ValueError:
                error(f'Could Not Convert {part}')
                
        listy.append(part)

    return listy[0]
            
            

def main():
    ''' Parse command line arguments and process expressions from standard
    input. '''

    try:
        if sys.argv[1] == '-h':
             usage()
    except IndexError:
        pass

    for line in sys.stdin:
        print(evaluate_expression(line.rstrip()))
        


# Main Execution

if __name__ == '__main__':
    main()

# vim: set sts=4 sw=4 ts=8 expandtab ft=python:
