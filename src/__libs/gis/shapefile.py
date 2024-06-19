import geopandas as gpd
from pyproj import CRS, Transformer
import pandas as pd

def analyzeShapeFileAt(inEPSG:int, filePath:str, coords, primaryIDs, columns, idColumnName:str):
    # Read shapefile
    gdf = gpd.read_file(filePath)

    # Transform to EPSG:2193
    gdf = gdf.to_crs(epsg=inEPSG)

    # Evaluate layers at sample locations
    point = gpd.points_from_xy(coords[0], coords[1], crs=CRS.from_epsg(inEPSG))
    res = pd.DataFrame(columns=columns, index=primaryIDs)
    for i,p in enumerate(point):
        polygon = gdf[gdf.geometry.contains(p)]
        if len(polygon) > 0:
            res.iloc[i,:] = polygon.loc[:,columns]

    return res


if __name__ == "__main__":

    shapeFile = "/Users/shabanil/Library/CloudStorage/OneDrive-VictoriaUniversityofWellington-STAFF/PhD/GIS/_Layers/Iris-SoilLayers/lris-nzlri-soil-SHP/nzlri-soil.shp"

    csvFile = pd.read_csv("/Users/shabanil/Library/CloudStorage/OneDrive-VictoriaUniversityofWellington-STAFF/PhD/__Data/NVS_OccurenceAndRange/1_TotalPlotInfo.csv", usecols=["ParentPlotID", "NorthingTM", "EastingTM"])
    csvFile.drop_duplicates(inplace=True)
    csvFile = csvFile[csvFile.ParentPlotID.isin([945,3213])]

    coordinates = (csvFile.EastingTM.to_numpy(), csvFile.NorthingTM.to_numpy())  # Replace with your coordinates

    #See Coordinate system here https://www.landcareresearch.co.nz/assets/Tools-And-Resources/Data/data-visualisation/Basic-GIS-manual.pdf

    # CRS definitions
    in_crs = 2193  # Replace with the CRS of your input coordinates


    value = analyzeShapeFileAt(2193, shapeFile, coordinates,csvFile.ParentPlotID,["SOIL","SOIL1"],"ParentPlotID")
    print(f"Value at coordinates {coordinates}: {value}")

