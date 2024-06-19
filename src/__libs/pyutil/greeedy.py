from typing import Callable

def solveMaxCoverage(allValues:set, allSets:set, setToValues:Callable, numSets:int):
    """Solves the max coverage problem for a given set of values and sets. Each set
    contains the values, setToValues functions transforms a set into the values it posesses."""

    if numSets == 0:
        return {'selectedSets': [],
                'uncoveredValues': allValues,
                'coveredValues': set()}

    #Find the set that covers the most values

    selectedSets = set()
    U = allValues
    for i in range(0,numSets):
        #find the species that has the biggest intersection with U
        bestSet = None
        bestOverlap = 0
        for s in allSets:
            Ai = setToValues(s)
            numOverlap = len(Ai.intersection(U))
            if numOverlap > bestOverlap:
                bestOverlap = numOverlap
                bestSet = s

        if bestSet is None: break #can't optimise further

        U = U - set(setToValues(bestSet))
        selectedSets.add(bestSet)
        allSets.remove(bestSet)

    return {'selectedSets': selectedSets,
            'uncoveredValues': U,
            'coveredValues': allValues-U}
