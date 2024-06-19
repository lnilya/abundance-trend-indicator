import pandas as pd
import numpy as np
import paths as PATHS
import src.__libs.pyutil as pyutil
import rasterio
from pyproj import Proj, transform
import numpy as np
import rasterio.fill as f
import warnings

def coordinatesNotInTIFF(geotiff_file, coordinates, in_crs, fallbackTifCRS = "EPSG:27200"):
    with rasterio.open(geotiff_file) as src:

        srcCRS = src.crs if src.crs is not None else fallbackTifCRS
        x, y = (coordinates[0], coordinates[1])

        if srcCRS != in_crs:

            #Suppress the deprecation warning, no clue how to fix it.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                src_proj, tgt_proj = Proj(init=in_crs), Proj(srcCRS)
                x, y = transform(src_proj, tgt_proj, coordinates[0], coordinates[1])

        # Find the pixel corresponding to the transformed coordinates
        row, col = src.index(x, y)
        row = np.array(row)
        col = np.array(col)
        #Identify which indicies need to be removed due to out of bounds
        outOfBounds = np.logical_or(np.logical_or(row < 0, col < 0), np.logical_or(row >= src.height, col >= src.width))

        return outOfBounds

def readGeoTiffAtCoordinates(geotiff_file, coordinates, primaryIDs, in_crs, nameOfColumn:str, fallbackCRS = "EPSG:27200", returnAsDF:bool = True, addRowAndCol:bool = False):
    """Loads a GeoTiff file and returns the value at the given coordinates. The coordinate system of the GeoTIFF is in its metadata, the other need to be specified.
    Result is returned as a pandas dataframe with the primaryIDs as index and the value at the coordinates as the new  column.

    """

    with rasterio.open(geotiff_file) as src:

        srcCRS = src.crs if src.crs is not None else fallbackCRS
        if srcCRS != in_crs:

            # Suppress the deprecation warning, no clue how to fix it.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Transform coordinates to match the CRS of the GeoTIFF
                src_proj, tgt_proj = Proj(init=in_crs), Proj(srcCRS)
                x, y = transform(src_proj, tgt_proj, coordinates[0], coordinates[1])
        else:
            x, y = (coordinates[0], coordinates[1])

        # Find the pixel corresponding to the transformed coordinates
        row, col = src.index(x, y)
        row = np.array(row)
        col = np.array(col)
        #Identify which indicies need to be removed due to out of bounds
        outOfBounds = np.logical_or(np.logical_or(row < 0, col < 0), np.logical_or(row >= src.height, col >= src.width))

        srcR = src.read(1)
        f.fillnodata(srcR, mask=src.read_masks(1), max_search_distance=4)
        noDataSpots = None
        if returnAsDF:
            res = pd.DataFrame({'ParentPlotID':primaryIDs})

            #Add some interpolation for the out of bounds values, this happens along coasts mostly.
            #Set the values that can be computed, all else remains NaN
            if nameOfColumn is not None:
                res.loc[~outOfBounds,nameOfColumn] = srcR[row[~outOfBounds], col[~outOfBounds]]
                noDataSpots = np.where(np.abs(res[nameOfColumn]) > 1000000)
            if addRowAndCol:
                res.loc[~outOfBounds,"mapX"] = col[~outOfBounds]
                res.loc[~outOfBounds,"mapY"] = row[~outOfBounds]
                res = res.astype({"mapX":int, "mapY":int})
                #cast to int

        else:
            #Initialize array with nans
            res = np.full(len(primaryIDs), np.nan)
            res[~outOfBounds] = srcR[row[~outOfBounds], col[~outOfBounds]]

            noDataSpots = np.where(np.abs(res) > 1000000)
            if addRowAndCol:
                mapx = np.full(len(primaryIDs), np.nan)
                mapy = np.full(len(primaryIDs), np.nan)
                mapy[~outOfBounds] = col[~outOfBounds]
                mapx[~outOfBounds] = row[~outOfBounds]
                res = (res, mapx, mapy)


        if noDataSpots is not None and  len(noDataSpots) > 1:
            print(f"Warning: {len(noDataSpots)} plots have no data for {nameOfColumn}.")


        return res




##JUST FOR TESTING

if __name__ == "__main__":

    geotiff_file = "/Users/shabanil/Library/CloudStorage/OneDrive-VictoriaUniversityofWellington-STAFF/PhD/GIS/_Layers/NZEnvDS_v1-1/final_layers_nzmg/solRad_winterPotential.tif"
    csvFile = pd.read_csv(PATHS.nvsTotalPlotInfo, usecols=["ParentPlotID", "NorthingTM", "EastingTM"])
    csvFile.drop_duplicates(inplace=True)


    coordinates = (csvFile.EastingTM.to_numpy(), csvFile.NorthingTM.to_numpy())  # Replace with your coordinates

    #See Coordinate system here https://www.landcareresearch.co.nz/assets/Tools-And-Resources/Data/data-visualisation/Basic-GIS-manual.pdf

    # CRS definitions
    in_crs = "EPSG:2193"  # Replace with the CRS of your input coordinates


    value = readGeoTiffAtCoordinates(geotiff_file, coordinates, csvFile.ParentPlotID, in_crs, "DistanceToCoast")
    print(f"Value at coordinates {coordinates}: {value}")

