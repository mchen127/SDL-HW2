# Problem 2

## Problem Statement
1. Download the dataset from https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip.
2. Train an LSTM network to predict future air temperature based on a sequence of past weather observations.
3. Present your results with appropriate plots and analysis.


## Dataset
| Index | Feature | Format | Description |
| --- | --- | --- | --- |
| 1 | Date Time | 01.01.2009 00:10:00 | Date-time reference |
| 2 | p (mbar) | 996.52 | Atmospheric pressure in millibars, a standard meteorological unit. |
| 3 | T (degC) | -8.02 | Air temperature in Celsius. |
| 4 | Tpot (K) | 265.4 | Potential temperature in Kelvin. |
| 5 | Tdew (degC) | -8.9 | Dew point temperature; the point at which air becomes saturated and water condenses. |
| 6 | rh (%) | 93.3 | Relative humidity, indicating how saturated the air is with water vapor. |
| 7 | VPmax (mbar) | 3.33 | Saturation vapor pressure. |
| 8 | VPact (mbar) | 3.11 | Actual vapor pressure. |
| 9 | VPdef (mbar) | 0.22 | Vapor pressure deficit. |
| 10 | sh (g/kg) | 1.94 | Specific humidity. |
| 11 | H2OC (mmol/mol) | 3.12 | Water vapor concentration. |
| 12 | rho (g/m ** 3) | 1307.75 | Air density. |
| 13 | wv (m/s) | 1.03 | Wind speed. |
| 14 | max. wv (m/s) | 1.75 | Maximum wind speed. |
| 15 | wd (deg) | 152.3 | Wind direction in degrees. |
