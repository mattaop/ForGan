# Forecasting Uncertainty for Univariate Time Series Using Generative Adversarial Networks
Release version 1.0: 26.06.2020

### Requirements
Dependencies necessary to run the code:
```bash
numpy
pandas
sklearn
scikit-image
tensorflow
keras
pydataset
pyyaml
statsmodels
tqdm
seaborn
pmdarima
```

Installation can be done manually or by creating an environment, activate it and install requirements:
```bash
conda create -n [name_of_environment] python=3.6
conda activate [name_of_environment]
pip install -r requirements.txt
```


### Data sets
In order to run the code, one have to download the data sets, and place it according to the following folder structure and name conventions:
```bash
|---- data  # Folder for data sets
|     |---- data_files  # Datasets requiered to train the models
|           |---- avocado.csv
|           |---- electricity.npy
|           |---- OsloTemperature.csv
``` 

The data sets can be found and downloaded here:

Avocado Price data set: https://www.kaggle.com/neuromusic/avocado-prices

Electricity consumption data set: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

Oslo Temperature data set: https://wiki.math.ntnu.no/lib/exe/fetch.php?tok=5deb8a&media=https%3A%2F%2Fwww.math.ntnu.no%2Femner%2FTMA4285%2F2019h%2Fpdf%2Fdata.xlsx

### Configurations
Set the path of the project folder in order to run the code:
```bash
|---- config  # Folder for configurations
|     |---- paths.yml   # Set project path
``` 
Change configurations in:
```bash
|---- config  # Folder for configurations
|     |---- config.yml   # Set configurations such as model, data set and other hyperparameters
``` 

### Run code
The following files have to be ran in order to do the experiments:

Baseline models:
```bash
time_series_pipeline_baseline.py
time_series_pipeline_avocado_baseline.py
```
Neural networks and generative adversarial networks:
```bash
time_series_pipeline_with_validation.py
time_series_pipeline_avocado.py
```
