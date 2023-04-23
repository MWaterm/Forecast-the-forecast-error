%% DAY AHEAD PREDICTIONS OF A QUANTITIES' FORECASTING ERROR TO IMPROVE THE FORECASTS
% DATA:     Excel-file with date (UTC), observations and forecasted values of the quantity 
%           National Holiday (GER)
% METHOD:   Using an decomposition model with an optional seasonal
%           component (hour of the week) and a stochastic component,
%           modelled with an time series model. The time series of forecast
%           errors is modelled in two different frameworks, a univariate
%           and a multivariate. All model frameworks are estimated for
%           every day-ahead forecast iteratively and use historical data
%           from different defined calibration windows for estimation; the
%           forecast is done as 24-hour-ahead forecast.
% OUTPUT:   Forecast of the forecast error of the quantity for every sub-model
%           Final improved point prediction of the quantity (combination of
%           the sub-models' forecasts)
%           Estimated parameters for evry forecasted day 
%           P-values of the estimated parameters for evry forecasted day 

clc
clear

%% Import configuration data
[filepath,xls_filename_data,xls_sheetname_data,xls_filename_Xdata,xls_sheetname_Xdata,xls_filename_holidays,xls_sheetname_holidays,rolling_window_lengths,date_format,ind_prices,ind_deseasonalisation, num_Xdata, savepath] = configuration_file()

%% Import the data

%holidays
opts = spreadsheetImportOptions("NumVariables", 1);
opts.Sheet = xls_sheetname_holidays;
opts.VariableNames = "Time";
opts.VariableTypes = "datetime";
opts = setvaropts(opts, "Time", "InputFormat", "");

holidayger = readtable([filepath, xls_filename_holidays], opts, "UseExcel", false);
clear opts


%actual and forecasted values
opts = spreadsheetImportOptions("NumVariables", 3);
opts.Sheet = xls_sheetname_data;
opts.VariableNames = ["time", "actual", "forecast"];
opts.VariableTypes = ["datetime", "double", "double"];
opts = setvaropts(opts, "time", "InputFormat", "");
dataimport = readtable([filepath, xls_filename_data], opts, "UseExcel", false);
dataimport(1,:) = [];
dataimport.time = dateshift(dataimport.time, 'start', 'hour', 'nearest'); 

dataimport.error = dataimport.actual - dataimport.forecast; 

%exogenous data (if wanted)
if num_Xdata == 0
    dataXimport = 0; 
elseif num_Xdata == 1
    opts = spreadsheetImportOptions("NumVariables", num_Xdata+1);
    opts.Sheet = xls_sheetname_Xdata;
    opts.VariableNames = ["time", "Xdata1"];
    opts.VariableTypes = ["datetime", "double"];
    opts = setvaropts(opts, "time", "InputFormat", "");
    dataXimport = readtable([filepath, xls_filename_Xdata], opts, "UseExcel", false);
    dataXimport(1,:) = [];
    dataXimport.time = dateshift(dataXimport.time, 'start', 'hour', 'nearest'); 
elseif num_Xdata == 2
    opts = spreadsheetImportOptions("NumVariables", num_Xdata+1);
    opts.Sheet = xls_sheetname_Xdata;
    opts.VariableNames = ["time", "Xdata1", "Xdata2"];
    opts.VariableTypes = ["datetime", "double", "double"];
    opts = setvaropts(opts, "time", "InputFormat", "");
    dataXimport = readtable([filepath, xls_filename_Xdata], opts, "UseExcel", false);
    dataXimport(1,:) = [];
    dataXimport.time = dateshift(dataXimport.time, 'start', 'hour', 'nearest'); 
end

%% Univariate model framework
    [prediction_uv] = model_uvXX_new(dataimport, dataXimport, holidayger, rolling_window_lengths, ind_prices, ind_deseasonalisation, date_format, savepath);
    writetable(prediction_uv,[savepath, 'prediction_uv.xlsx']);


%% Multivariate model framework
    [prediction_mv] = model_mvXX_new(dataimport, dataXimport, holidayger, rolling_window_lengths, ind_prices, ind_deseasonalisation, date_format, savepath);
    writetable(prediction_mv,[savepath, 'prediction_mv.xlsx']);


%% Combination of both model frameworks and the different rolling window lengths (= combination of all individual sub-models)
prediction_submodels = [prediction_uv, prediction_mv(:,5:end)]; 
ind_startpred = find(isnan(prediction_submodels(:,5)) == 0, 1, 'first');
prediction_submodels = prediction_submodels(ind_startpred:end, :); 

num_calibration_days = max(rolling_window_lengths/24); 


prediction_submodels.nonw_comb = mean(table2array(prediction_submodels(:,5:end)),2); 
timevec = [prediction_submodels.time(1):1/24:prediction_submodels.time(1)+hours(size(prediction_submodels,1)-1)]'; 


%% Regression fit und Regression forecast of the weights - calculation of optimal weights
period_calibration = calendarDuration(0,0,num_calibration_days);
if ind_prices == 1
    numperiods_Dayahead = 24; 
    start_train = prediction_submodels.time(25); 
    end_train = prediction_submodels.time(25) + period_calibration - hours(1); 
    start_pred = prediction_submodels.time(25) + period_calibration; 
    start_train_r = 25; 
else
   numperiods_Dayahead = 48; 
   start_train = prediction_submodels.time(1); 
   end_train = prediction_submodels.time(1) + period_calibration - hours(1); 
   start_pred = prediction_submodels.time(25) + period_calibration; 
   start_train_r = 1; 
end

[~, end_train_r] = max(prediction_submodels.time == end_train);
[~ ,start_pred_r] = max(prediction_submodels.time == start_pred);
time_pred = prediction_submodels.time(start_pred_r:end);

predictions = table2array(prediction_submodels(:,5:end-1));
eps = prediction_submodels.error;
eps_predtime = eps(start_pred_r:end); 
eps_avcom = prediction_submodels.nonw_comb(start_pred_r:end); 

modelfit_h = zeros(size(time_pred, 1), 2); 
modelcoefficients_h = zeros(size(time_pred, 1), 2*n); 
modelcoefficients_pvalue_h = zeros(size(time_pred, 1), 2*n);
regforecast_h = zeros(size(time_pred)); 



for t = 1:length(time_pred)/24
    for h = 0:23
        time_h = prediction_submodels.time(start_train_r:end_train_r); 
        predictions_h = predictions(start_train_r:end_train_r, :);
        predictions_h = predictions_h(hour(time_h) == h,:);
        eps_h = eps(start_train_r:end_train_r);
        eps_h = eps_h(hour(time_h) == h,:);
        time_pred_h = time_pred(start_pred_r-(num_calibration_days+1)*24:start_pred_r+23-(num_calibration_days+1)*24);
        predictions_pred_h = predictions(start_pred_r:start_pred_r+23,:);
        predictions_pred_h = predictions_pred_h(hour(time_pred_h) == h,:);

        modelcoefficients_h(t*24+1-24+h, :) = lsqnonneg(predictions_h, eps_h); 
        regforecast_h(t*24+1-24+h) = predictions_pred_h*modelcoefficients_h(t*24+1-24+h, :)';
    end
    
    
    start_train = start_train+days(1);
    end_train = end_train+days(1);
    start_pred = start_pred+days(1);
    
    start_train_r = start_train_r + 24; 
    end_train_r = end_train_r + 24; 
    start_pred_r = start_pred_r + 24; 
    
end


%% Error measures of the combined forecasts
time_pred_orig = time_pred; 
% if ind_prices == 1
%     ind_pred_date = find(year(time_pred) == 2018, 1, 'first'); 
% else
%     ind_pred_date = find(year(time_pred) == 2021, 1, 'first'); 
% end
ind_pred_date = 1; 
time_pred = time_pred(ind_pred_date:end);


% Error of the final forecasts by a combination with optimal weights,
% estimated with an hourly regression model
errors_regh = [eps_predtime - regforecast_h]; 
errors_regh = errors_regh(ind_pred_date:end); 
errors_regh(:, end+1) = errors_regh(:,1).^2; 
errors_regh(:, end+1) = abs(errors_regh(:,1));

MSEregh = nanmean(errors_regh(:,2));
RMSEregh = sqrt(MSEregh); 
MAEregh = nanmean(errors_regh(:,3)); 

% yearwise
year_pred = year(time_pred); 
[years_forecasted, ia, ic] = unique(year_pred); 
MSEregh_yearly = accumarray(ic, errors_regh(:, 2),[], @mean); 
MAEregh_yearly = accumarray(ic, errors_regh(:, 3),[], @mean); 
RMSEregh_yearly = sqrt(MSEregh_yearly); 
error_measure_regh = [RMSEregh MAEregh; RMSEregh_yearly MAEregh_yearly]; 


% Error measures of the initial forcast
errors_initial = eps_predtime(ind_pred_date:end); 
errors_initial(:, end+1) = errors_initial(:,1).^2; 
errors_initial(:, end+1) = abs(errors_initial(:,1));
    
MSEi = nanmean(errors_initial(:,2));
RMSEi = sqrt(MSEi); 
MAEi = nanmean(errors_initial(:,3)); 

% yearwise
year_pred = year(time_pred); 
[years_forecasted, ia, ic] = unique(year_pred); 
MSEi_yearly = accumarray(ic, errors_initial(:, 2),[], @mean); 
MAEi_yearly = accumarray(ic, errors_initial(:, 3),[], @mean); 
RMSEi_yearly = sqrt(MSEi_yearly);    
error_measure_i = [RMSEi MAEi; RMSEi_yearly MAEi_yearly]; 


% Error of the final forecasts by a combination with the arithmetic mean
errors_avcom = eps_predtime - eps_avcom; 
errors_avcom = errors_avcom(ind_pred_date:end); 
errors_avcom(:, end+1) = errors_avcom(:,1).^2; 
errors_avcom(:, end+1) = abs(errors_avcom(:,1));
    
MSEavcom = nanmean(errors_avcom(:,2));
RMSEavcom = sqrt(MSEavcom); 
MAEavcom = nanmean(errors_avcom(:,3)); 

% yearwise
year_pred = year(time_pred); 
[years_forecasted, ia, ic] = unique(year_pred); 
MSEavcom_yearly = accumarray(ic, errors_avcom(:, 2),[], @mean); 
MAEavcom_yearly = accumarray(ic, errors_avcom(:, 3),[], @mean); 
RMSEavcom_yearly = sqrt(MSEavcom_yearly);    
error_measure_avcom = [RMSEavcom MAEavcom; RMSEavcom_yearly MAEavcom_yearly]; 

predictions = predictions(start_pred_r:end);
if RMSEavcom <= RMSEregh
    print('The average of all sub-models point predictions is the combined and final point prediction for the forecast error to improve the quantities forecast.')
    prediction = prediction_submodels; 
    writetable(prediction,[savepath, 'prediction_forecasterror.xlsx']);
else    
    print('The weighted combination of all sub-models point predictions is the combined and final point prediction for the forecast error to improve the quantities forecast.')
    prediction = [eps_predtime(ind_pred_date:end) predictions(ind_pred_date:end) eps_avcom(ind_pred_date:end)]; 
    writematrix(prediction,[savepath, 'prediction_forecasterror.xlsx']);
end




