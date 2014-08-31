import sys
import time

def p(t):
    #print variable and type of it
    print('__')
    print(t)
    print(type(t))

def pe(t):
    #print variable, type of it, and finally exits the program
    print('__')
    print(t)
    print(type(t))
    sys.exit()

def launch_time():
    #starts both real and computational time
    t = [time.clock(), time.time()]
    return t

def print_time(t):
    #measure and prints both real and computational time
    return [time.clock()-t[0], time.time()-t[1]]

def start_time():
    t = time.time()
    return t

def get_time(t):
    T = time.time()-t
    return T
