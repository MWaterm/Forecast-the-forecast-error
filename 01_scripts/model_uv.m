function [prediction] = model_uv(data, dataX, holidays, rollingwindows, ind_prices, ind_deseasonalisation, date_format, savepath)
% Implementation of the univariate model framework to forcast the forecast
% error
num_ex = size(dataX,2)-1; 
numparams = 6+3+num_ex;

rolling_window_lengths_str = num2str(rollingwindows); 
rolling_window_lengths_max = max(rollingwindows); 

startprediction = datetime(data.time(1)+calendarDuration(0,0,0,rolling_window_lengths_max+24,0,0),'Inputformat',date_format);
endtraining = startprediction - calendarDuration(0,0,0,1,0,0); 

%% prediction allocation
ind = find((year(startprediction)==year(data.time))&(month(startprediction)==month(data.time))&(day(startprediction)==day(data.time))&(hour(startprediction)==hour(data.time))); 
prediction = data(ind:end, :); 
prediction.time(end+1:end+24) = data.time(end) + calendarDuration(0,0,0,1,0,0) : hours(1) : data.time(end) + calendarDuration(0,0,0,24,0,0); 
data_forecast = prediction; 

for m = 1:length(rollingwindows)
%%
    prediction.(rolling_window_lengths_str(m,:)) = NaN(size(prediction,1),1);

    PValues = zeros(size(prediction,1)/24-1, numparams+1);
    Values = PValues;
    
starttraining = endtraining - calendarDuration(0,0,0,rollingwindows(m)-1,0,0);
% Initialisation
startpred = startprediction
endpred = startprediction + hours(48) 
starttrain = starttraining
endtrain = endtraining
daybeforetrain = starttrain - calendarDuration(0,0,1)

for i = 1:size(prediction,1)/24-1
%%
    i_starttrain = find((year(starttrain)==year(data.time))&(month(starttrain)==month(data.time))&(day(starttrain)==day(data.time))&(hour(starttrain)==hour(data.time))); 
    i_endtrain = find((year(endtrain)==year(data.time))&(month(endtrain)==month(data.time))&(day(endtrain)==day(data.time))&(hour(endtrain)==hour(data.time))); 
    i_startpred = find((year(startpred)==year(data.time))&(month(startpred)==month(data.time))&(day(startpred)==day(data.time))&(hour(startpred)==hour(data.time)));  
    i_endpred = find((year(endpred)==year(data.time))&(month(endpred)==month(data.time))&(day(endpred)==day(data.time))&(hour(endpred)==hour(data.time))); 
    
        data_train = data(i_starttrain-24:i_endtrain, :); 
    data_train.deseas = data_train.error; 
    if ind_deseasonalisation == 0
        [regressormat, howmeansmat] = regressors(data_train,holidays,'uv');
        regressormat(1:24,:) = []; 
        howmeansmat(1:24,:) = []; 
        data_train(1:24,:) = []; 
    else
        [~, howmeansmat] = regressors(data_train,holidays,'uv');
        data_train.deseas = data_train.error - sum(howmeansmat,2);
        [regressormat, ~] = regressors(data_train,holidays,'uv');
        regressormat(1:24,:) = []; 
        howmeansmat(1:24,:) = []; 
        data_train(1:24,:) = []; 
    end
    
    iholidays = sum(regressormat(:,169:192),2);
    if num_ex == 0
        regressormatX = [regressormat iholidays]; 
    elseif num_ex == 1
        dataX_train = dataX(i_starttrain:i_endtrain, :); 
        dataX_forecast = dataX(i_endtrain+1:i_endtrain+48, :);
        regressormatX = [regressormat dataX_train.Xdata1 iholidays];
    elseif num_ex == 2
        dataX_train = dataX(i_starttrain:i_endtrain, :); 
        dataX_forecast = dataX(i_endtrain+1:i_endtrain+48, :);
        regressormatX = [regressormat dataX_train.Xdata1 dataX_train.Xdata2 iholidays]; 
    end
    
    data_forecast.deseas = data_forecast.error; 
    pred_reg = [data_train(end-23:end,:); data_forecast(i*24-23:i*24+24,:)];
    if ind_deseasonalisation == 0
        [regressormat_pred, ~] = regressors(pred_reg,holidays,'uv');
        regressormat_pred(1:24,:) = []; 
    else
        [~, howmeansmat_pred] = regressors(pred_reg,holidays,'uv');
        pred_reg.deseas = pred_reg.error - sum(howmeansmat_pred,2);
        [regressormat_pred, ~] = regressors(pred_reg,holidays,'uv');
        regressormat_pred(1:24,:) = []; 
        pred_reg(1:24,:) = []; 
    end
    
    iholidays_pred = sum(regressormat_pred(:,169:192),2);
    if num_ex == 0
        regressormatX_pred = [regressormat_pred iholidays_pred]; 
    elseif num_ex == 1
        regressormatX_pred = [regressormat_pred dataX_forecast.Xdata1 iholidays_pred]; 
    elseif num_ex == 2
        regressormatX_pred = [regressormat_pred dataX_forecast.Xdata1 dataX_forecast.Xdata2 iholidays_pred]; 
    end
    
    ARIMA = arima('ARLags',[1,2,24,168], 'MALags', [1]);
    [EstMdl_Time_Series_loop, EstParamCov_loop, logL_loop, info_loop] = estimate(ARIMA, data_train.deseas(ARIMA.P+1:end,:), 'X', regressormatX(:,193:end), 'Display', 'off');
    Yfo_loop = data_train.deseas(end-ARIMA.P+1:end,:);
    [Y_loop, YMSE_loop] = forecast(EstMdl_Time_Series_loop, 48, Yfo_loop, 'XF', regressormatX_pred(:,193:end));
    Results_loop = summarize(EstMdl_Time_Series_loop);
	PValues_loop = Results_loop.Table.PValue;
    PValues(i, :) = PValues_loop'; 
    Values(i, :) = info_loop.X'; 
    
    if ind_prices == 1
        wd_pred = weekdayholiday(prediction.time(i*24-23), holidays);
        if ((isempty(find(1 == regressormat(:,169)))==1) && (wd_pred == 8))
            wd_pred = 7; 
        end
        how_pred = wd_pred*24-23:wd_pred*24; 
        pred_seas = sum(howmeansmat(:,how_pred),2);
        pred_seas = nonzeros(pred_seas); 
        if size(pred_seas,1) > 24
            pred_seas(25:end) = []; 
        end
        if ind_deseasonalisation == 1
            prediction.(rolling_window_lengths_str(m,:))(i*24-23:i*24) = pred_seas + Y_loop(1:24);
        else
            prediction.(rolling_window_lengths_str(m,:))(i*24-23:i*24) = Y_loop(1:24);
        end
    else
        wd_pred = weekdayholiday(prediction.time(i*24+1), holidays);
        if ((isempty(find(1 == regressormat(:,169)))==1) && (wd_pred == 8))
            wd_pred = 7; 
        end
        how_pred = wd_pred*24-23:wd_pred*24; 
        pred_seas = sum(howmeansmat(:,how_pred),2);
        pred_seas = nonzeros(pred_seas); 
        if size(pred_seas,1) > 24
            pred_seas(25:end) = []; 
        end
        if ind_deseasonalisation == 1
            prediction.(rolling_window_lengths_str(m,:))(i*24+1:i*24+24) = pred_seas + Y_loop(25:48);
        else
            prediction.(rolling_window_lengths_str(m,:))(i*24+1:i*24+24) = Y_loop(25:48);
        end
    end
    
    %One day forward
    startpred = startpred + hours(24);
    endpred = endpred + hours(24); 
    starttrain = starttrain + hours(24); 
    endtrain = endtrain + hours(24); 
    disp(['Next step: Rolling window: ', rolling_window_lengths_str(m,:), ', start prediction: ', datestr(startpred)])
end

% Writing of estimated parameters and th p-values of the estimation for
% every forecasted day and every rolling window
% writematrix(prediction.time(hour(prediction.time) == 0), [savepath, 'Prediction_uvxx_pvalues_', num2str(rollingwindows(m)), '.xlsx'], 'Range','A2');
% writematrix(PValues,[savepath, 'Prediction_uvxx_pvalues_', num2str(rollingwindows(m)), '.xlsx'], 'Range','B2');
% writematrix(prediction.time(hour(prediction.time) == 0), [savepath, 'Prediction_uvxx_values_', num2str(rollingwindows(m)), '.xlsx'], 'Range','A2');
% writematrix(Values,[savepath, 'Prediction_uvxx_values_', num2str(rollingwindows(m)), '.xlsx'], 'Range','B2');
end
