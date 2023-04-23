function [wd_inchol] = weekdayholiday(date, holidays)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
wd_inchol = weekday(date); 
feiertag_sum = 0; 
    for t = 1:length(holidays.Time)
%        if ~(((day(holidays.Time(t))==24) && (month(holidays.Time(t))==12)) || ((day(holidays.Time(t))==31) && (month(holidays.Time(t))==12)))
            i_feiertag = find((day(date) == day(holidays.Time(t))) & month(date) == month(holidays.Time(t)) & year(date) == year(holidays.Time(t)));
            wd_inchol(i_feiertag) = 8; 
            feiertag_sum = feiertag_sum + length(i_feiertag); 
%        end
    end
end

