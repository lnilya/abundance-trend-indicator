
from colorama import Fore, Back, Style


def chapPrint(x):
    """Prints a heading. Green background, white text"""
    print(Back.LIGHTGREEN_EX + x + Style.RESET_ALL)

def successPrint(x):
    print(Fore.GREEN + x + Style.RESET_ALL)
def infoPrint(x):
    print(Fore.BLUE + x + Style.RESET_ALL)

def errorPrint(x):
    print(Back.LIGHTRED_EX + x + Style.RESET_ALL)

def endPrint(x):
    """Prints an ending. Green text and a new line"""
    print(Fore.GREEN + x + Style.RESET_ALL)
    print("")