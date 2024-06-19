import re
import os
from typing import List, Union, Dict


def getAllFilesWithSubfolders(folder:str, pattern:str = None)->Dict[str,Union[str,List[str]]]:
    """More advanced version of getAllFiles that searches in subfolders as well. This is done by using the find command on the command line."""

    #escape spaces in folder
    folder = folder.replace(" ", "\ ")
    if folder.endswith("/"): folder = folder[:-1]
    if not pattern.startswith("/"): pattern = "/" + pattern
    if pattern.startswith("*"):
        res = os.popen(f"find {folder} -type f -path '{pattern}'").read()
    else:
        res = os.popen(f"find {folder} -type f -path '*{pattern}'").read()

    files = res.split("\n")
    #regex from pattern
    patternRE = "(.*)" + pattern.replace("*", "([^/]*)")
    patternRE = re.compile(patternRE)
    matched_files = {}
    for f in files:
        if f == "": continue
        #math pattern and extract the parts that match the *
        match = patternRE.search(f)
        if match:
            matched_files[f] = list(match.groups())[1:]

    return matched_files

if __name__ == "__main__":
    getAllFilesWithSubfolders("/Users/shabanil/Documents/Uni/DeslippeLab/efti/_v3 link/_Models/Classifiers",
                 "*_combined/*_SVM_*_AdultsOnly.pickle")

def getAllFiles(folder:str, pattern:str = None)->Dict[str,Union[str,List[str]]]:
    """
    @deprecated

    Retrieves all files in the given folder that match the given pattern.

    :param folder: The folder path where the files should be searched in.
    :param pattern: The pattern to match the file names against. '*' represents zero or more characters. If None, the pattern will be the last part of the folder path, allowing to pass the pattern in the folder.
    :return: A dictionary containing the matched file names as keys and the parts of the file names that matched the star in the pattern as values.
    """

    if pattern is None:
        f = folder.split("/")
        pattern = f[-1]
        folder = "/".join(f[:-1])


    #Replace the dots inside the regex pattern
    regex_pattern = re.sub(r'\.', r'\\.', pattern)
    regex_pattern = re.sub(r'\+', r'\\+', pattern)
    # Create a regular expression pattern from the given pattern
    # by replacing '*' with '(.*)' to capture the part that matches the star
    regex_pattern = re.sub(r'\*', r'(.*)', regex_pattern)

    if folder[-1] != "/": folder += "/"

    compiled_pattern = re.compile(regex_pattern)

    all_files = os.listdir(folder)
    matched_files = {}

    for file in all_files:
        match = compiled_pattern.search(file)
        if match:
            # Store the file and the part of the file name that
            # matched the star in the pattern
            matched_files[folder + file] = list(match.groups())


    #If pattern contains only one * the items of the result should not be a list
    if pattern.count("*") == 1:
        matched_files = {k:v[0] for k,v in matched_files.items()}


    return matched_files