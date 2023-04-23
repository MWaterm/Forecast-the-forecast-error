"""
This script calculates the probabilistic forecasts of the forecast error by Quantile Regression Averaging. 
It uses the point predictions of the individual sub-models, which are calculated in the file "forecast_forecasterror_mainfile.mat". 

INPUT: The point predictions need to be transformed to an Excel-file with the following eight columns: 
        - date: timestamp in UTC
        - error: forecast error of the initial quantitie's forecast
        - uv_1: point prediction of the forecast error, estimated with univariate sub-model and the first rolling window size
        - uv_2: point prediction of the forecast error, estimated with univariate sub-model and the second rolling window size
        - uv_3: point prediction of the forecast error, estimated with univariate sub-model and the third rolling window size
        - mv_1: point prediction of the forecast error, estimated with multivariate sub-model and the first rolling window size
        - mv_2: point prediction of the forecast error, estimated with multivariate sub-model and the second rolling window size
        - mv_3: point prediction of the forecast error, estimated with multivariate sub-model and the third rolling window size

OUTPUT: CSV-file (probabilistic_prediction.csv) with the probabilistic forecasts of the foreacst error
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import statsmodels.formula.api as smf
from sklearn.metrics import mean_pinball_loss


#######################################################################################################################################################################
#######################################################################################################################################################################
# ---- FUNCTIONS ----
#######################################################################################################################################################################
#######################################################################################################################################################################

def PredIntervalCalculator(data, len_train_data, quantiles):

    start_date_data = data[data.columns[0]].iloc[0]
    start_date_preds = start_date_data + timedelta(hours=len_train_data)

    pred_intervals = pd.DataFrame(data[data[data.columns[0]] >= start_date_preds][data.columns[0]]).reset_index().drop(columns='index')
    pred_intervals['error'] = pd.DataFrame(data[data[data.columns[0]] >= start_date_preds][data.columns[3]]).reset_index().drop(columns='index')
    col_names = ['pred_interval_len_' + str(h) for h in quantiles]
    pred_intervals[col_names] = np.NaN

    for i in range(0, len(pred_intervals), 24):
        for tau in quantiles:

            data_train_test = data.iloc[i:len_train_data + i + 24]
            pred_intervals['pred_interval_len_' + str(tau)].iloc[i:i + 24] = QRA(data_train_test, tau)

        print(1)
    return pred_intervals


def QRA(data_train_test, tau):

    mod = smf.quantreg("error ~ uv_1 + uv_2 + uv_3 + mv_1 + mv_2 + mv_3", data_train_test[['error', 'uv_1', 'uv_2', 'uv_3', 'mv_1', 'mv_2', 'mv_3']].iloc[0:len(data_train_test)-24])
    pred = mod.fit(q=tau).predict(data_train_test[['uv_1', 'uv_2', 'uv_3', 'mv_1', 'mv_2', 'mv_3']].iloc[len(data_train_test)-24:len(data_train_test)])

    return pred

def QuantilePredictionEvaluator(predictions, quantile_levels, var_name):

    avg_pinball_loss = pd.DataFrame(np.zeros((5088,21)), columns=list(map(str, quantiles)))

    for q in quantile_levels:
        avg_pinball_loss[str(q)] = mean_pinball_loss(predictions[var_name], predictions['pred_interval_len_' + str(q)], alpha=q)
        
    avg_pinball_loss_per_quantile = avg_pinball_loss[list(map(str, quantiles))].mean(axis = 0)
    avg_pinball_loss_overall = avg_pinball_loss_per_quantile.mean()

    return avg_pinball_loss, avg_pinball_loss_per_quantile, avg_pinball_loss_overall



def QuantilePredictionEvaluatorPerHourAndWeekday(predictions, quantile_levels, var_name):

    # CRPS Approx. per Hour

    avg_pinball_loss_qra_per_hour = pd.DataFrame(np.zeros((24, 1)).T, columns=list(range(0, 24)))
    avg_pinball_loss_q = pd.DataFrame(np.zeros((len(list(map(str, quantiles))), 1)).T, columns=list(map(str, quantiles)))
    avg_pinball_loss_qra_per_hour_and_quantile = pd.DataFrame(np.zeros((24, len(list(map(str, quantiles))))).T, columns=list(range(0, 24)), index= list(map(str, quantiles)))

    for h in avg_pinball_loss_qra_per_hour.columns:

        predictions_h = predictions[predictions['time'].dt.hour == h]

        for q in quantile_levels:
            avg_pinball_loss_q[str(q)] = mean_pinball_loss(predictions_h[var_name],
                                                           predictions_h['pred_interval_len_' + str(q)], alpha=q)

        avg_pinball_loss_per_quantile = avg_pinball_loss_q[list(map(str, quantiles))].mean(axis=0)
        avg_pinball_loss_qra_per_hour_and_quantile.loc[:,h] = avg_pinball_loss_per_quantile
        avg_pinball_loss_qra_per_hour[h] = avg_pinball_loss_per_quantile.mean()


    # CRPS Approx. per Weekday

    avg_pinball_loss_qra_per_weekday = pd.DataFrame(np.zeros((7, 1)).T, columns=list(range(0, 7)))
    avg_pinball_loss_q = pd.DataFrame(np.zeros((len(list(map(str, quantiles))), 1)).T, columns=list(map(str, quantiles)))
    avg_pinball_loss_qra_per_weekday_and_quantile = pd.DataFrame(np.zeros((7, len(list(map(str, quantiles))))).T, columns=list(range(0, 7)), index= list(map(str, quantiles)))

    for d in avg_pinball_loss_qra_per_weekday.columns:

        predictions_h = predictions[predictions['time'].dt.dayofweek == d]

        for q in quantile_levels:
            avg_pinball_loss_q[str(q)] = mean_pinball_loss(predictions_h[var_name],
                                                           predictions_h['pred_interval_len_' + str(q)], alpha=q)

        avg_pinball_loss_per_quantile = avg_pinball_loss_q[list(map(str, quantiles))].mean(axis=0)
        avg_pinball_loss_qra_per_weekday_and_quantile.loc[:,d] = avg_pinball_loss_per_quantile
        avg_pinball_loss_qra_per_weekday[d] = avg_pinball_loss_per_quantile.mean()


    return avg_pinball_loss_qra_per_hour, avg_pinball_loss_qra_per_weekday, avg_pinball_loss_qra_per_hour_and_quantile, avg_pinball_loss_qra_per_weekday_and_quantile








#######################################################################################################################################################################
#######################################################################################################################################################################
# ---- Code ----
#######################################################################################################################################################################
#######################################################################################################################################################################
"""
Define variables and parameters
"""

file_path = 'C:/Users/Prueba/Documents/Python Scripts/ErrorImprovement/data/'

xls_filename_pp = '.xlsx'       # filename of the Excel-file containing all necessary input data for Quantile Regression Averaging
xls_sheetname_inputQRA = 'inputQRA'     # sheetname of the input data needed in this script

quantity = ''   # e.g., price, load, wind

quantiles = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975]     # list of quantiles which are estimated for the foreacast error by Quantile Regression Averaging
len_train_data_qra = 8736   # length of calibration window to estimate the parameters of the quantile regession averaging methodology


"""
Quantile Regression Averaging to calculate probabilistic forecasts
"""
preds = pd.read_excel((file_path + xls_filename_pp), sheet_name = xls_sheetname_inputQRA)

pred_intervals = PredIntervalCalculator(preds, len_train_data = len_train_data_qra, quantiles = quantiles)
avg_pinball_loss_qra, avg_pinball_loss_per_quantile_qra, avg_pinball_loss_overall_qra = QuantilePredictionEvaluator(pred_intervals, quantile_levels = quantiles, var_name='error')
    
pred_intervals.to_csv(file_path + 'probabilistic_prediction.csv', sep = ';')

avg_pinball_loss_qra['overall'] = avg_pinball_loss_overall_qra
avg_pinball_loss_qra.to_csv(file_path + 'avg_pinball_loss_probabilistic_prediction.csv', sep = ';', index = False)
    

