from typing import Dict,List
import pandas as pd
import json
import pickle
def writePickle(outpath:str, data:any, prefix:str, silent=False):

    if outpath[-6:] != 'pickle':
        print("Warning in writing Pickle File: outpath does not end in pickle. Replacing the extension with pickle instead")
        outpath = ".".join(outpath.split(".")[:-1] + ['pickle'])

    pickle.dump(data, open(outpath, "wb"))

    if not silent:
        print("[%s] Written Pickle data to %s" % (prefix, outpath.split("/")[-1]))

def writeJSON(outpath:str, data:Dict, prefix:str):
    """
    Writes a dictionary to a json file and prints a confirmation
    :param outpath: The location to write the JSON file to
    :param data: The data as a dictionary
    :param prefix: The prefix for the confirmation print
    :return: None
    """

    if outpath[-4:] != 'json':
        print("Warning in writing JSON File: outpath does not end in json. Replacing the extension with json instead")
        outpath = ".".join(outpath.split(".")[:-1] + ['xlsx'])

    json_object = json.dumps(data, indent=4)

    with open(outpath, 'w') as fp:
        fp.write(json_object)

    print("[%s] Written JSON data to %s" % (prefix, outpath.split("/")[-1]))


def writePandasToCSV(df:pd.DataFrame, outpath:str, message:str = None, printFun = print, **toCSVargs):
    """Helper with writing Pandas to CSV. Prints a confirmation messages and checks if path ends in CSV."""
    if outpath[-3:] != 'csv':
        print("Warning in writing CSV File: outpath does not end in csv. Replacing the extension with csv instead")
        outpath = ".".join(outpath.split(".")[:-1] + ['csv'])

    if len(toCSVargs) == 0:
        df.to_csv(outpath, index=False)
    else:
        df.to_csv(outpath, **toCSVargs)

    printConfirmation(df,outpath,message,printFun=printFun)
def writeExcelWithSheets(outpath:str,dataframes:Dict[str,pd.DataFrame],comment:List[str] = None):
    """
    Writes a dictionary of dataframes to an excel file, each dataframe on a separate sheet.
    :param outpath: Pathfor the excel file
    :param dataframes: Dataframes to write to the excel file
    :param comment: if not None, a comment will be added as a separate sheed. Each entry in the list is a separate line
    :return:
    """

    if outpath[-4:] != 'xlsx':
        print("Warning in writing Excel File: outpath does not end in xlsx. Replacing the extension with xlsx instead")
        outpath = ".".join(outpath.split(".")[:-1] + ['xlsx'])

    if comment is not None:
        dataframes['Comments'] = pd.DataFrame( {'Comments': comment})

    # Create an Excel writer using pandas
    writer = pd.ExcelWriter(outpath, engine='xlsxwriter')

    # Write the DataFrame to the first sheet
    for sheetName,df in dataframes.items():
        df.to_excel(writer, sheet_name=sheetName, index=False)

    # Save the Excel file˘
    writer.close()


def printConfirmation(df:pd.DataFrame, path:str, prefix:str = None, printFun = print):
    if df is None: ldf = -1
    else: ldf = len(df)
    if prefix is None:
        printFun("Written %d rows to %s"%(ldf,path.split("/")[-1]))
    else:
        printFun("[%s] Written %d rows to %s"%(prefix, ldf,path.split("/")[-1]))
