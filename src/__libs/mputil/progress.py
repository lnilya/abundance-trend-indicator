from typing import Callable, List
import multiprocessing
import tqdm
from tqdm import tqdm

def _wrapper(args):
    #Check if is dictionary or list
    if isinstance(args,dict):
        func = args.pop('func')
        return func(**args)
    else:
        func = args[0]
        return func(*args[1:])

def runParallel(func:Callable, args:List,poolSize:int=11, debug=False, progressMessage:str = None):
    """
    Runs a function in parallel with a progress bar
    :param func: Function to run
    :param args: List of arguments to pass to the function, either as named dictionary arguments or as a numbered list.
    :param poolSize: Pool size Number of processes to use
    :param debug: If true, run in debug mode (no parallelization, all runs in main process)
    :return: List of results Whatever the function returns
    """

    #Add func to args
    if isinstance(args[0], dict):
        args = [dict(func=func,**arg) for arg in args]
    else:
        args = [(func,*arg) for arg in args]

    results = []
    if debug:
        for a in args:
            results.append(_wrapper(a))
    else:
        pbar = tqdm(total=len(args), desc=progressMessage)
        with multiprocessing.Pool(processes=poolSize) as pool:
            for result in pool.imap(_wrapper, args):
                results.append(result)
                pbar.update(1)

    print("") #Clear line
    return results