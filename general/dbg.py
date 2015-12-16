import time


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
