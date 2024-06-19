from typing import List

from src.classes.Serializable import Serializable


class VariableList(Serializable):
    """Stores a list of variables used of predicting. Essentially a list with aname. The name is used to store and identify datasets. Lists can be added together to create new datsets."""

    name:str = ""
    list:List[str] = []

    @staticmethod
    def fromDict(dict)->"VariableList":
        return VariableList(dict["name"],dict["list"])

    def toDict(self):
        return {"name":self.name,"list":self.list}

    def __init__(self, name:str, list:List[str]):
        self.list = list
        self.name = name

    def __len__(self):
        return len(self.list)
    def __add__(self, other):
        if other is None: return self

        if isinstance(other, str):
            other = [other]

        if isinstance(other, VariableList):
            return VariableList(self.name + "_" + other.name, self.list + list(set(other.list) - set(self.list)))
        elif isinstance(other, list):
            return VariableList(self.name+f"_plus_{len(other)}", list(set(self.list+other)))

    def _addName(self,addedName:str):
        nameList:List[str] = self.name.split("_") + [addedName]
        #sort alphabetically and join
        nameList.sort()
        self.name = "_".join(nameList)

    def __iadd__(self, other):
        if other is None: return
        if isinstance(other, VariableList):
            self.list += list(set(other.list) - set(self.list)) #Add only new variables!
            self._addName(other.name)
        elif isinstance(other, list):
            self.list += other
        else:
            raise ValueError("Can only add VariableList to VariableList")
    def __iter__(self):
        return iter(self.list)
