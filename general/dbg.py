import sys
import time


def p(t):
    """ Print variable and type of it """
    print('__')
    print(t)
    print(type(t))

def pe(t):
    """ Print variable, type of it, and finally exits the program """
    print('__')
    print(t)
    print(type(t))
    sys.exit()

def start_time():
    """ Starts both real and computational time """
    t = [time.clock(), time.time()]
    return t

def get_time(t):
    """ Measure and prints both real and computational time """
    return [time.clock()-t[0], time.time()-t[1]]


