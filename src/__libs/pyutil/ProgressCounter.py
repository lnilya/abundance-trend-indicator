from time import time
from datetime import datetime

class ProgressCounter:
    totalSteps:int
    curStep:int
    lastTrace:int
    traceInterval:int
    format: str

    def __init__(self, total:int, format:str = 'Progress %.1f %% (Remaining: %s)', traceInterval:int = 10):
        self.totalSteps = total
        self.format = format
        self.curStep = 0
        self.lastTrace = 0
        self.traceInterval = traceInterval if traceInterval > 0 else 1
        self.startTime = time()
        self.print()

    def __mod__(self, other):
        return self.curStep % other

    def print(self):
        elapsedTime = time() - self.startTime
        if self.curStep > 0:
            timeLeft = (self.totalSteps - self.curStep) * (elapsedTime / self.curStep)
            #Parse time into readable format
            timeLeft = datetime.utcfromtimestamp(timeLeft).strftime('%H:%M:%S')
        else:
            timeLeft = "-"

        print(self.format % (100 * self.curStep / self.totalSteps, timeLeft))

    def done(self, msg:str = 'Finished'):
        print(msg)

    def inc(self,bysteps = 1, silent:bool = False):
        self.curStep += bysteps
        if (self.curStep - self.lastTrace) >= self.traceInterval:
            if not silent: self.print()
            self.lastTrace = self.curStep - (self.curStep % self.traceInterval)
