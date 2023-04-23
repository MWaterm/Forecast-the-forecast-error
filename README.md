# Forecast the forecast error: Improving point forecasts and generating density forecasts in energy markets

### Mira Watermeyer, Franziska Scheller

Forecast errors of variables in the energy markets contain information, patterns and structures which can be captured and modelled. We provide a flexible approach to predict the forecast error independently of the initial forecast model and the target variable through point and probabilistic forecasts, thus improving initial forecasts. Our methodology is based on ARMA models and quantile regression averaging. We average over arbitrary hyperparameters, such as rolling window sizes, so that the approach becomes robust to random missteps by potential users. We demonstrate the framework's effectiveness using three examples: Electricity price forecasts, load forecasts, and wind forecasts.


### Links: 
tba

### Explanations: 

In folder "Scripts" users can find all implementations needed to forecast a forecast error.  

To run the application, define all configurations (filepaths, filenames, model choice) in the file "configuration_file.mat". 

The file "forecast_forecasterror_mainfile.mat" forecasts the forecast error. It calculates the point predictions of the individual sub-models and combines them as defined in the belonging work. All other files written in Matlab are included automatically. 

The file "probabilistic_forecast_forecasterror.py" contains the calculation of probabilistic forecasts of the forecast error in addition to the estimated point forecast. It uses the results of the script "forecast_forecasterror_mainfile.mat". 


### Citing IntEG

The model published in this repository is free: you can access, modify and share it under the terms of the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. This model is shared in the hope that it will be useful for further research on topics of risk-aversion, investments, flexibility and uncertainty in ectricity markets but without any warranty of merchantability or fitness for a particular purpose. 

If you use the model or its components for your research, we would appreciate it if you
would cite us as follows:
```
The reference to the working paper version is as follows:

  M. Watermeyer and F. Scheller (2023), Forecast the forecast error: Improving point forecasts and generating density forecasts in energy markets, Working paper
```
