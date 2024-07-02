# Abundance Trend Indicator 

This library uses various machine learning models to learn under which environmental conditions a species increases or decreases its abundance (classification problem).

The details can be found in the accompanying paper: ...add link...

## Installation

1. Check out the repository or clone it to your local machine.

2. Set up a virtual environment (venv) 

```bash
python3 -m venv venv
```
3. Activate the virtual environment

```bash
source venv/bin/activate
```
4. Install the required packages

```bash
pip install -r requirements.txt
```
**Note:** We strongly recommend using an IDE like Pycharm instead of using the command line. Especially activating the virtual environment can be automated.

5. Unpack the demo data to the _data folder in the repository

Download the data from ... 
Unpack to _data folder inside the repository

## Input Data

This repository provides some synthetic data to demonstrate the usage of the library. 

There are three essential datasets you need to provide in order to run the library:
- Site Data
- Predictor Variables
- Occurrence Data


### 1. Site Data (_data/_input/PlotInfo.csv)
This is a CSV file containing information about forest inventory plots. With the columns `PlotID,ObservationID,Year,mapX,mapY`
- **PlotID** is a unique ID for each plot
- **ObservationID** is a unique ID for each survey of the plot. The model requires plots remeasured at least twice. These plots would have the same PlotID but different ObservationID and Years.
- **Year** is the year of the survey
- **mapX** and **mapY** are the coordinates of the plot. To avoid any projection errors we use pixel coordinates referring to the geotif files provided as predictor variables. Convert your plot locations to pixel coordinates before use.
 

### 2. Predictor Variables (_data/_input/_predictor_variables/**/*.tif)
This is a set of geotiff layers that represent the environmental conditions. All tifs must be in the same resolution. The projection does not matter as we use pixel coordinates.

Layers that do not change over time (e.g. elevation) are stored in a tif file. The name of the file will be used as the name of the variable throughout the entire analysis. These layers are referred to internally as "GeoLayers"

Layers that change over time (e.g. annual mean temperature) are stored in a folder. The name of the folder will be used as the name of the variable throughout the entire analysis. The folder contains tif files named as the year at which they were measured. These layers are referred to internally as "ClimateLayers"

The library will work regardless of the number or type of layers you provide.

### 3. Occurrence Data (_data/_input/Occurrences.csv)
This is a CSV file containing forest inventory data all species with the columns `PlotID,ObservationID,Species,NumIndividuals,Total Diameter`.

- **PlotID** is a unique ID for each plot. Same IDs as in the Site Data.
- **ObservationID** is a unique ID for each survey of the plot. Same IDs as in the Site Data.
- **Species** is the name of the species.
- **NumIndividuals** is the number of individuals of the species at the site (PlotID) and survey (ObservationID).
- **Total Diameter** is the sum of the diameters of all individuals of the species at the site (PlotID) and survey (ObservationID). This column can be set to 0 if the data is not available. It is only used if the number of individuals does not change and we still want to estiamte the abundance change (referred to as **Abundance and DBH** dataset in the paper)

### Synthetic data for testing

The ... archive contains the synthetic data used in the paper. Unzip it into the repository's _data folder for testing. It contains parts real and parts synthetic data for testing. 

1. Site data
The site data contains a random subset of real sites from the New Zealand Vegetation survey but the years and ObservationID are synthetic. The only years present in the data are 2000 and 2019. The observation IDs are identical with the PlotIDs for the year 2000 and set to the negative value for the year 2019.
2. Predictor variables
The geo layers are taken from [NZENVDS](https://datastore.landcareresearch.co.nz/ne/dataset/nzenvds) and the climate layers are generated from [HOTRUNZ](https://essd.copernicus.org/articles/14/2817/2022/essd-14-2817-2022-discussion.html).
These values contain real data.
3. Occurrence data 
The occurrence data is synthetically generated for an "Elevation Species" that shifts along a yhypothetical elevation gradient. It is generated using the following procedure: 
- First estimate the elevational distribution of all plots in the site data and estimating obtain its mean M and standard deviation S. 
- Then generate two gaussian presence probability distributions "2000" and "2019". The 2000 distribution is set to μ = M - 0.5*S and the 2019 distribution is set to μ = M + 0.5*S. Both distributions have a standard deviation of σ = S*0.3.    
- A subset of plots where the species is assumed present is then drawn from each distribution generating the occurrence data.

The resulting elevational distribution of the synthetic species is shown in the figure below.

![Elevation Species](./SyntheticDistribution.jpg)



Since we are not allowed to publish the raw data used in the paper, we provide synthetic data to demonstrate the usage of the library. The data is stored in the _data folder in the repository. 


### Usage

The model goes through a series of steps to first preprocess the data into the necessary format, then train the models and finally make predictions. 
Each step generates a set of files that are stored in the _data folder. The results 




