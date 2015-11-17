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

class Timer():
    def __init__(self, name='time', start=True):
        self.name = name
        if start:
            self.start()

    def start(self):
        self.vals = []
        self.ttin = [time.clock(), time.time()]

    def measure(self, print_time=True):
        self.vals.append([time.clock()-self.ttin[0], time.time()-self.ttin[1]])
        if print_time:
            print self

    def __repr__(self):
        return 'time (%s): %s' % (self.name, str(self.vals))
