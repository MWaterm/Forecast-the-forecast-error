function [regressormat, howmeansmat] = regressors(data_pred,holidays, model)

wd_inchol = weekday(data_pred.time); 
feiertag_sum = 0; 
    for t = 1:length(holidays.Time)
            i_feiertag = find((day(data_pred.time) == day(holidays.Time(t))) & month(data_pred.time) == month(holidays.Time(t)) & year(data_pred.time) == year(holidays.Time(t)));
            wd_inchol(i_feiertag) = 8; 
            feiertag_sum = feiertag_sum + length(i_feiertag); 
    end

id_ymd = year(data_pred.time)*10000+month(data_pred.time)*100+day(data_pred.time);
id_how = wd_inchol*100+hour(data_pred.time); 

[ymd, ia_ymd, ic_ymd] = unique(id_ymd); 
[how, ia_how, ic_how] = unique(id_how); 
how = [1:192]'; 

means_ymd = accumarray(ic_ymd, data_pred.deseas, [], @mean); 
min_ymd = accumarray(ic_ymd, data_pred.deseas, [], @min); 
max_ymd = accumarray(ic_ymd, data_pred.deseas, [], @max); 
means_how = accumarray(ic_how, data_pred.deseas, [], @mean); 

means_ymd = means_ymd(ic_ymd); 
min_ymd = min_ymd(ic_ymd); 
max_ymd = max_ymd(ic_ymd); 
means_how = means_how(ic_how); 
how = how(ic_how); 

howdummymat = zeros(length(data_pred.time), 192); 
howmeansmat = zeros(length(data_pred.time), 192); 
for t = 1:192
    howdummymat(how == t,t) = 1; 
    howmeansmat(how == t,t) = means_how(how == t); 
end

%Shift due to d-1 data needed for d 
means_ymd(25:end) = means_ymd(1:end-24); 
means_ymd(1:24) = NaN; 
min_ymd(25:end) = min_ymd(1:end-24); 
min_ymd(1:24) = NaN; 
max_ymd(25:end) = max_ymd(1:end-24); 
max_ymd(1:24) = NaN; 

%Define h-1
actualh_1b = data_pred.forecast; 
actualh_1b(2:end,1) = data_pred.deseas(1:end-1); 
actualh_1b(1,1) = data_pred.deseas(24); 

if strcmp(model,'uv') == 1
    regressormat = [howdummymat, min_ymd, max_ymd]; 
elseif strcmp(model,'mv') == 1
    regressormat = [howdummymat, min_ymd, max_ymd, actualh_1b]; 
end

end