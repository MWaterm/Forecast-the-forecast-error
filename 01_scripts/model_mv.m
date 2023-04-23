function [prediction] = model_mv(data, dataX, holidays, rollingwindows, ind_prices, ind_deseasonalisation, date_format, savepath)
% Implementation of the multivariate model framework to forcast the
% forecast error

rolling_window_lengths_str = num2str(rollingwindows); 
rolling_window_lengths_max = max(rollingwindows); 

startprediction = datetime(data.time(1)+calendarDuration(0,0,0,rolling_window_lengths_max+24,0,0),'Inputformat',date_format);
endtraining = startprediction - calendarDuration(0,0,0,1,0,0); 

num_ex = size(dataX,2)-1; 
numparams = 4+3+num_ex+1; 
structname = {'h0'; 'h1'; 'h2'; 'h3'; 'h4'; 'h5'; 'h6'; 'h7'; 'h8'; 'h9'; 'h10'; 'h11'; 'h12'; 'h13'; 'h14'; 'h15'; 'h16'; 'h17'; 'h18'; 'h19'; 'h20'; 'h21'; 'h22'; 'h23'};

%% prediction allocation
ind = find((year(startprediction)==year(data.time))&(month(startprediction)==month(data.time))&(day(startprediction)==day(data.time))&(hour(startprediction)==hour(data.time))); 
prediction = data(ind:end, :); 
prediction.time(end+1:end+24) = data.time(end) + calendarDuration(0,0,0,1,0,0) : hours(1) : data.time(end) + calendarDuration(0,0,0,24,0,0); 
data_forecast = prediction; 

for m = 1:length(rollingwindows)
%%
    prediction.(rolling_window_lengths_str(m,:)) = NaN(size(prediction,1),1);

    PValues = struct();
    Values = PValues;
    for h = 0:23
        PValues.(char(structname(h+1,1))) = zeros(size(prediction,1)/24-1, numparams+1);
        Values.(char(structname(h+1,1))) = zeros(size(prediction,1)/24-1, numparams+1);
    end
    
starttraining = endtraining - calendarDuration(0,0,0,rollingwindows(m)-1,0,0);
% Initialisation
startpred = startprediction
endpred = startprediction + hours(48) 
starttrain = starttraining
endtrain = endtraining
daybeforetrain = starttrain - calendarDuration(0,0,1)
estMdl_h = struct();
Yfo_loop2 = struct();

for i = 1:size(prediction,1)/24-1  
    i_starttrain = find((year(starttrain)==year(data.time))&(month(starttrain)==month(data.time))&(day(starttrain)==day(data.time))&(hour(starttrain)==hour(data.time))); 
    i_endtrain = find((year(endtrain)==year(data.time))&(month(endtrain)==month(data.time))&(day(endtrain)==day(data.time))&(hour(endtrain)==hour(data.time))); 
    i_startpred = find((year(startpred)==year(data.time))&(month(startpred)==month(data.time))&(day(startpred)==day(data.time))&(hour(startpred)==hour(data.time)));  
    i_endpred = find((year(endpred)==year(data.time))&(month(endpred)==month(data.time))&(day(endpred)==day(data.time))&(hour(endpred)==hour(data.time))); 
    
    data_train = data(i_starttrain-24:i_endtrain, :); 
    data_train.deseas = data_train.error; 
    if ind_deseasonalisation == 0
        [regressormat, howmeansmat] = regressors(data_train,holidays,'mv');
        regressormat(1:24,:) = []; 
        howmeansmat(1:24,:) = []; 
        data_train(1:24,:) = []; 
    else
        [~, howmeansmat] = regressors(data_train,holidays,'mv');
        data_train.deseas = data_train.error - sum(howmeansmat,2);
        [regressormat, ~] = regressors(data_train,holidays,'mv');
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
        [regressormat_pred, ~] = regressors(pred_reg,holidays,'mv');
        regressormat_pred(1:24,:) = []; 
    else
        [~, howmeansmat_pred] = regressors(pred_reg,holidays,'mv');
        pred_reg.deseas = pred_reg.error - sum(howmeansmat_pred,2);
        [regressormat_pred, ~] = regressors(pred_reg,holidays,'mv');
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
    end
    
    data_trainh = table(); 
    data_trainh.time = data_train.time(hour(data_train.time) == 0); 
    for h = 0:23
        namee = ['e',num2str(h)];
        data_trainh.(namee) = data_train.deseas(hour(data_train.time) == h); 

        ARIMA = arima('ARLags',[1,2,7]);
        [EstMdl_Time_Series_loop, EstParamCov_loop, logL_loop, info_loop] = estimate(ARIMA, data_trainh.(namee)(ARIMA.P+1:end,:), 'Y0', data_trainh.(namee)(1:ARIMA.P,:), 'X', regressormatX(hour(data_train.time) == h,193:end), 'Display', 'off');
        estMdl_h.(namee) = EstMdl_Time_Series_loop;
        
        Yfo_loop = data_trainh.(namee)(end-ARIMA.P+1:end,:);
        regressormatX_pred_loop = regressormatX_pred(hour(prediction.time(i*24-23:i*24+24)) == h,193:end);
        [Y_loop1, YMSE_loop1] = forecast(EstMdl_Time_Series_loop, 1, Yfo_loop, 'XF', regressormatX_pred_loop(1,:));
        
        Yfo_loop2.(namee) = [Yfo_loop(2:end, :); Y_loop1]; 
        if h == 23
            regressormatX_pred((hour(prediction.time(i*24-23:i*24)) == 0), end-num_ex-1) = Y_loop1; % prediction of actual hour must be exogenous variable for next hour
        else
            regressormatX_pred((hour(prediction.time(i*24-23:i*24)) == h+1), end-num_ex-1) = Y_loop1; % prediction of actual hour must be exogenous variable for next hour
        end
        Results_loop = summarize(EstMdl_Time_Series_loop);
        PValues_loop = Results_loop.Table.PValue;
        PValues.(char(structname(h+1,1)))(i, :) = PValues_loop'; 
        Values.(char(structname(h+1,1)))(i, :) = info_loop.X';        
        
        if ind_prices == 1
            if ind_deseasonalisation == 1
                prediction.(rolling_window_lengths_str(m,:))(i*24-23+h) = pred_seas(h+1) + Y_loop1;
            else
                prediction.(rolling_window_lengths_str(m,:))(i*24-23+h) = Y_loop1;
            end
        end
    end
    
    if ind_prices ~= 1
        for h = 0:23
            namee = ['e',num2str(h)];
%             data_predh.(namee) = data_pred.deseas(hour(data_pred.time) == h); 
            
            EstMdl_Time_Series_loop = estMdl_h.(namee); 
            Yfo_loop = Yfo_loop2.(namee); 
            regressormatX_pred_loop = regressormatX_pred(hour(prediction.time(i*24-23:i*24+24)) == h,193:end);
            [Y_loop2, YMSE_loop2] = forecast(EstMdl_Time_Series_loop, 1, Yfo_loop, 'XF', regressormatX_pred_loop(2,:));
            
            if h == 23
                regressormatX_pred((hour(prediction.time(i*24-23:i*24+24)) == 0), end-num_ex-1) = Y_loop2; % prediction of actual hour must be exogenous variable for next hour
            else
                regressormatX_pred((hour(prediction.time(i*24-23:i*24+24)) == h+1), end-num_ex-1) = Y_loop2; % prediction of actual hour must be exogenous variable for next hour
            end
            if ind_deseasonalisation == 1
                prediction.(rolling_window_lengths_str(m,:))(i*24+1+h) = pred_seas(h+1) + Y_loop2;
            else
                prediction.(rolling_window_lengths_str(m,:))(i*24+1+h) = Y_loop2;
            end
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
% for h = 0:23
%     sheetname = ['Sheet_h', num2str(h)]; 
%     writematrix(prediction.time(hour(prediction.time) == 0), [savepath, 'Prediction_mvxx_pvalues_', num2str(rollingwindows(m)), '.xlsx'], 'Sheet', sheetname, 'Range','A2');
%     writematrix(PValues.(char(structname(h+1,1))),[savepath, 'Prediction_mv_pvalues_', num2str(rollingwindows(m)), '.xlsx'], 'Sheet', sheetname, 'Range','B2');
%     writematrix(prediction.time(hour(prediction.time) == 0), [savepath, 'Prediction_mv_values_', num2str(rollingwindows(m)), '.xlsx'], 'Sheet', sheetname, 'Range','A2');
%     writematrix(Values.(char(structname(h+1,1))),[savepath, 'Prediction_mvxx_values_', num2str(rollingwindows(m)), '.xlsx'], 'Sheet', sheetname, 'Range','B2');
% end

    
end
end

