function [filepath,xls_filename_data,xls_sheetname_data,xls_filename_Xdata,xls_sheetname_Xdata,xls_filename_holidays,xls_sheetname_holidays,rolling_window_lengths,date_format,ind_prices,ind_deseasonalisation,num_Xdata, savepath] = configuration_file()
% Configuration file to set all necessary parameters for modelling and
% forecasting the forecast error, done with the
% "forecast_forecasterror_mainfile" script

% path of all files
filepath = 'C:\Users\Prueba\Documents\error improvement\matlab\data\'


%file- and sheetname of the actual observations and the initial forecasts
%of the quantity
xls_filename_data = '.xlsx' % Excel file containing columns date, real_observation, prediction in that order
xls_sheetname_data = '' % Sheet name of point forecast excel file

%file- and sheetname of exogenous variables (if not needed, leave out)
xls_filename_Xdata = '.xlsx' % Excel file containing columns date, real_observation, prediction in that orde
xls_sheetname_Xdata = 'exogenous' % Sheet name of point forecast excel file

%file- and sheetname containing the national holidays (list with dates)
xls_filename_holidays = 'holiday_germany.xlsx' % Excel file containing holidays
xls_sheetname_holidays = 'holiday_nation' % Sheet name of holiday excel file



% path of all resulting files (especially the forecasts of the forecast
% errors)
savepath = 'C:\Users\Prueba\Documents\error improvement\matlab\load results\'



% Modelling settings
rolling_window_lengths = [7416, 8088, 8736]' % in hours

ind_prices = 0  % is the forecasted quantity a day-ahead price? (1 = yes, 0 = no)
num_Xdata = 0   % how many exogenous data should be included in the model frameworks? (0 = none, otherwise 1-2)
ind_deseasonalisation = 1   % should the models include a deterministic seasonal component? (1 = yes, 0 = no)


date_format = '%Y-%m-%d %H:%M:%S'

end

