import time

startTimes = []

def tic(printMsg: str = None):
    """Measures times passed. call tic() before start and toc() when finished to get a trace of elapsed time."""
    global startTimes
    startTimes += [time.time()]
    if printMsg is not None:
        print(printMsg)

def tocr(msg: str = 'Ellpased Time: %.2fms'):
    global startTimes
    se = startTimes.pop()
    end = time.time()
    return (1000 * (end - se))


def toc(task: str = None, divideBy: float = 1.0, printFun = print):
    global startTimes
    se = startTimes.pop()
    end = time.time()
    if task is None:
        msg = 'Ellpased Time: %.2f ms'
    else:
        msg = 'Ellpased Time for ' + task + ': % .2f ms'

    p = msg % (1000 * (end - se) / divideBy)
    if printFun is not None:
        printFun(p)
    return p
