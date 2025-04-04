#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: norhaansaadawi
"""


#Opening the dataset
import pandas as pd
data = pd.read_csv('Dataset.csv')


#Viewing the data
print(data.columns)
print(data)
print(len(data))
data.describe()


#Setting date/time as the index
data['Date/Time (UTC)'] = pd.to_datetime(data['Date/Time (UTC)'])
data = data.sort_values(by="Date/Time (UTC)")
data.set_index('Date/Time (UTC)', inplace=True)


#Keeping exact winter season data
# Want to delete values:
#For 2019 from March 20 9:58pm to Dec 22 4:14am 
#For 2020 from March 20 3:49am to Dec 21 10:02am
#For 2021 from March 20 9:37 am to Dec 21 3:59pm
#For 2022 from March 20 3:33pm to Dec 21 9:48pm
#For 2023 from March 20 9:24pm to Dec 22 3:27am

start2019 = pd.to_datetime('2019-03-20 21:58:00')
end2019 = pd.to_datetime('2019-12-22 04:14:00')

start2020 = pd.to_datetime('2020-03-20 03:49:00')
end2020 = pd.to_datetime('2020-12-21 10:02:00')

start2021 = pd.to_datetime('2021-03-20 09:37:00')
end2021 = pd.to_datetime('2021-12-21 15:59:00')

start2022 = pd.to_datetime('2022-03-20 15:33:00')
end2022 = pd.to_datetime('2022-12-21 21:48:00')

start2023 = pd.to_datetime('2023-03-20 21:24:00')
end2023 = pd.to_datetime('2023-12-22 03:27:00')

datatodrop = data.query('(@start2019 < index < @end2019) or '
                        '(@start2020 < index < @end2020) or '
                        '(@start2021 < index < @end2021) or '
                        '(@start2022 < index < @end2022) or '
                        '(@start2023 < index < @end2023)')

print(datatodrop)

data = data.drop(datatodrop.index)

print(data)


#Calculating missing values and addressing them
nan_count = data.isna().sum()
print(nan_count)


data['Dew Point Temp (°C)'].interpolate(method='time', inplace=True)
data['Temp (°C)'].interpolate(method='time', inplace=True)
data['Wind Spd (km/h)'].interpolate(method='time', inplace=True)
data['Rel Hum (%)'].interpolate(method='time', inplace=True)
data['Precip. Amount (mm)'].interpolate(method='time', inplace=True)
data['Wind Dir (10s deg)'].interpolate(method='time', inplace=True)
data['Stn Press (kPa)'].interpolate(method='time', inplace=True)

# Checking to see if any temperature is > 20°C to see if humidex formula needs to be applied for filling missing values based on Government of Canada site
# Also checking if temperature is <= 0 or wind speed is > 0 for wind chill formula according to Government of Canada site
print(data['Temp (°C)'].describe())
print(data['Wind Spd (km/h)'].describe())


# filling missing wind chill as per formula on Government of Canada site
import numpy as np

def calculate_wind_chill(row):
    "Calculates the wind chill to indicate how cold the weather feels to the average person"
    "2 formulas provided by the Government of Canada"
    
    T = row['Temp (°C)']
    v = row['Wind Spd (km/h)']

    if pd.notna(row['Wind Chill']):
    
        return row['Wind Chill']

    if T <= 0 and v > 0:
        if v >= 5:
            # first formula
            WCI = 13.12 + 0.6215 * T - 11.37 * (v ** 0.16) + 0.3965 * T * (v ** 0.16)
        else:
            # second formula
            WCI = T + (v / 5) * (13.12 + 0.6215 * T - 13.12)
        return WCI
    else:
        # Wind Chill not applicable
        return np.nan


# Applying to all dataframe rows
data['Wind Chill'] = data.apply(calculate_wind_chill, axis=1)

nan_count = data.isna().sum()
print(nan_count)



#Removing unecessary columns and ones that contain too many missing values and can't be filled
columns_to_drop = ['Longitude (x)', 'Latitude (y)', 'Climate ID', 'Station Name', 'Temp Flag', 'Dew Point Temp Flag', 'Rel Hum Flag', 'Precip. Amount Flag', 'Wind Dir Flag', 'Wind Spd Flag',
                  'Visibility (km)', 'Visibility Flag', 'Stn Press Flag', 'Hmdx', 'Hmdx Flag', 'Wind Chill Flag', 'Weather']

data.drop(columns=columns_to_drop, inplace=True)

#Dropping missing values that can't be filled in from remaining columns
data = data.dropna()
nan_count = data.isna().sum()
print(nan_count)


#Identifying which timestamps are covered in the training set 2019-2022
#Creating an index including all timestamps to be converted to the training data

first_winter2019 = pd.date_range(start='2019-01-01 00:00:00', end='2019-03-20 21:00:00', freq='H')
second_winter2019 = pd.date_range(start='2019-12-22 05:00:00', end='2019-12-31 23:00:00', freq='H')

first_winter2020 = pd.date_range(start='2020-01-01 00:00:00', end='2020-03-20 03:00:00', freq='H')
second_winter2020 = pd.date_range(start='2020-12-21 11:00:00', end='2020-12-31 23:00:00', freq='H')

first_winter2021 = pd.date_range(start='2021-01-01 00:00:00', end='2021-03-20 09:00:00', freq='H')
second_winter2021 = pd.date_range(start='2021-12-21 16:00:00', end='2021-12-31 23:00:00', freq='H')

first_winter2022 = pd.date_range(start='2022-01-01 00:00:00', end='2022-03-20 15:00:00', freq='H')
second_winter2022 = pd.date_range(start='2022-12-21 22:00:00', end='2022-12-31 23:00:00', freq='H')

full_index = first_winter2019.append([second_winter2019,first_winter2020,second_winter2020,first_winter2021,second_winter2021,first_winter2022,second_winter2022])

print(full_index)


print(len(full_index))

data_2019_to_2022 = data.query('Year >= 2019 and Year <= 2022')

print(data_2019_to_2022)
print(len(data_2019_to_2022))


full_index_mdh = full_index.strftime('%m-%d-%H').unique()
data_mdh = data_2019_to_2022.index.strftime('%m-%d-%H').unique()

print(full_index_mdh)

print(data_mdh)

print(len(full_index_mdh))
  
full_index_mdh_set = set(full_index_mdh)
data_mdh_set = set(data_mdh)

missing_mdh = full_index_mdh.difference(data_mdh)
print(missing_mdh)

print('missing data for:',missing_mdh)
print('missing data for:',len(missing_mdh),'timestamps')



#Exploratory Data Analysis

#identifying correlations between temp and variables

import matplotlib.pyplot as plt
import seaborn as sns

print(data.columns)

numeric = ['Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)',
                    'Precip. Amount (mm)', 'Wind Spd (km/h)', 'Stn Press (kPa)',
                     'Wind Dir (10s deg)','Wind Chill']

corr_matrix = data[numeric].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# EDA showing daily,hourly,monthly averages for temp, dew point temp, and wind chill per year
# EDA showing time series for temp, dew point, and wind chill over the 4 years

print(data)

year2019 = data.query('Year == 2019')
print(year2019)

year2020 = data.query('Year == 2020')
print(year2020)

year2021 = data.query('Year == 2021')
print(year2021)

year2022 = data.query('Year == 2022')
print(year2022)

year2023 = data.query('Year == 2023')
print(year2023)




#Monthly averages for wind chill
monthly_trend2019 = year2019.groupby('Month')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
monthly_trend2019.plot(kind='bar', color = 'brown')
plt.title('Average Wind Chill Per Month 2019')
plt.xlabel('Month')
plt.ylabel('Wind Chill')
plt.show()



monthly_trend2020 = year2020.groupby('Month')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
monthly_trend2020.plot(kind='bar', color = 'red')
plt.title('Average Wind Chill Per Month 2020')
plt.xlabel('Month')
plt.ylabel('Wind Chill')
plt.show()



monthly_trend2021 = year2021.groupby('Month')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
monthly_trend2021.plot(kind='bar', color = 'green')
plt.title('Average Wind Chill Per Month 2021')
plt.xlabel('Month')
plt.ylabel('Wind Chill')
plt.show()



monthly_trend2022 = year2022.groupby('Month')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
monthly_trend2022.plot(kind='bar', color = 'orange')
plt.title('Average Wind Chill Per Month 2022')
plt.xlabel('Month')
plt.ylabel('Wind Chill')
plt.show()



monthly_trend2023 = year2023.groupby('Month')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
monthly_trend2023.plot(kind='bar', color = 'purple')
plt.title('Average Wind Chill Per Month 2023')
plt.xlabel('Month')
plt.ylabel('Wind Chill')
plt.show()



#daily averages for wind chill

daily_trend2019 = year2019.groupby('Day')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
daily_trend2019.plot(kind='bar', color = 'brown')
plt.title('Average Wind Chill Per Day 2019')
plt.xlabel('Day')
plt.ylabel('Wind Chill')
plt.show()

daily_trend2020 = year2020.groupby('Day')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
daily_trend2020.plot(kind='bar', color = 'red')
plt.title('Average Wind Chill Per Day 2020')
plt.xlabel('Day')
plt.ylabel('Wind Chill')
plt.show()

daily_trend2021 = year2021.groupby('Day')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
daily_trend2021.plot(kind='bar', color = 'green')
plt.title('Average Wind Chill Per Day 2021')
plt.xlabel('Day')
plt.ylabel('Wind Chill')
plt.show()

daily_trend2022 = year2022.groupby('Day')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
daily_trend2022.plot(kind='bar', color = 'orange')
plt.title('Average Wind Chill Per Day 2022')
plt.xlabel('Day')
plt.ylabel('Wind Chill')
plt.show()


daily_trend2023 = year2023.groupby('Day')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
daily_trend2023.plot(kind='bar', color = 'purple')
plt.title('Average Wind Chill Per Day 2023')
plt.xlabel('Day')
plt.ylabel('Wind Chill')
plt.show()

#hourly averages for wind chill

hourly_trend2019 = year2019.groupby('Time (UTC)')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
hourly_trend2019.plot(kind='bar', color = 'brown')
plt.title('Average Wind Chill By Time (UTC) 2019')
plt.xlabel('Time (UTC)')
plt.ylabel('Wind Chill')
plt.show()



hourly_trend2020 = year2020.groupby('Time (UTC)')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
hourly_trend2020.plot(kind='bar', color = 'red')
plt.title('Average Wind Chill By Time (UTC) 2020')
plt.xlabel('Time (UTC)')
plt.ylabel('Wind Chill')
plt.show()



hourly_trend2021 = year2021.groupby('Time (UTC)')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
hourly_trend2021.plot(kind='bar', color = 'green')
plt.title('Average Wind Chill By Time (UTC) 2021')
plt.xlabel('Time (UTC)')
plt.ylabel('Wind Chill')
plt.show()



hourly_trend2022 = year2022.groupby('Time (UTC)')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
hourly_trend2022.plot(kind='bar', color = 'orange')
plt.title('Average Wind Chill By Time (UTC) 2022')
plt.xlabel('Time (UTC)')
plt.ylabel('Wind Chill')
plt.show()



hourly_trend2023 = year2023.groupby('Time (UTC)')['Wind Chill'].mean()
plt.figure(figsize=(8, 4))
hourly_trend2023.plot(kind='bar', color = 'purple')
plt.title('Average Wind Chill By Time (UTC) 2023')
plt.xlabel('Time (UTC)')
plt.ylabel('Wind Chill')
plt.show()



#Monthly averages for dew point temp
monthly_trend = year2019.groupby('Month')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'brown')
plt.title('Average Dew Point Temp (°C) Per Month 2019')
plt.xlabel('Month')
plt.ylabel('Dew Point Temp (°C)')
plt.show()



monthly_trend = year2020.groupby('Month')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'red')
plt.title('Average Dew Point Temp (°C) Per Month 2020')
plt.xlabel('Month')
plt.ylabel('Dew Point Temp (°C)')
plt.show()



monthly_trend = year2021.groupby('Month')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'green')
plt.title('Average Dew Point Temp (°C) Per Month 2021')
plt.xlabel('Month')
plt.ylabel('Dew Point Temp (°C)')
plt.show()



monthly_trend = year2022.groupby('Month')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'orange')
plt.title('Average Dew Point Temp (°C) Per Month 2022')
plt.xlabel('Month')
plt.ylabel('Dew Point Temp (°C)')
plt.show()



monthly_trend = year2023.groupby('Month')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'purple')
plt.title('Average Dew Point Temp (°C) Per Month 2023')
plt.xlabel('Month')
plt.ylabel('Dew Point Temp (°C)')
plt.show()



#daily averages for dew point temp

daily_trend = year2019.groupby('Day')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'brown')
plt.title('Average Dew Point Temp (°C) Per Day 2019')
plt.xlabel('Day')
plt.ylabel('Dew Point Temp (°C)')
plt.show()

daily_trend = year2020.groupby('Day')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'red')
plt.title('Average Dew Point Temp (°C) Per Day 2020')
plt.xlabel('Day')
plt.ylabel('Dew Point Temp (°C)')
plt.show()

daily_trend = year2021.groupby('Day')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'green')
plt.title('Average Dew Point Temp (°C) Per Day 2021')
plt.xlabel('Day')
plt.ylabel('Dew Point Temp (°C)')
plt.show()

daily_trend = year2022.groupby('Day')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'orange')
plt.title('Average Dew Point Temp (°C) Per Day 2022')
plt.xlabel('Day')
plt.ylabel('Dew Point Temp (°C)')
plt.show()


daily_trend = year2023.groupby('Day')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'purple')
plt.title('Average Dew Point Temp (°C) Per Day 2023')
plt.xlabel('Day')
plt.ylabel('Dew Point Temp (°C)')
plt.show()

#hourly averages for dew point temp

hourly_trend = year2019.groupby('Time (UTC)')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'brown')
plt.title('Average Dew Point Temp (°C) By Time (UTC) 2019')
plt.xlabel('Time (UTC)')
plt.ylabel('Dew Point Temp (°C)')
plt.show()

hourly_trend = year2020.groupby('Time (UTC)')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'red')
plt.title('Average Dew Point Temp (°C) By Time (UTC) 2020')
plt.xlabel('Time (UTC)')
plt.ylabel('Dew Point Temp (°C)')
plt.show()

hourly_trend = year2021.groupby('Time (UTC)')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'green')
plt.title('Average Dew Point Temp (°C) By Time (UTC) 2021')
plt.xlabel('Time (UTC)')
plt.ylabel('Dew Point Temp (°C)')
plt.show()

hourly_trend = year2022.groupby('Time (UTC)')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'orange')
plt.title('Average Dew Point Temp (°C) By Time (UTC) 2022')
plt.xlabel('Time (UTC)')
plt.ylabel('Dew Point Temp (°C)')
plt.show()

hourly_trend = year2023.groupby('Time (UTC)')['Dew Point Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'purple')
plt.title('Average Dew Point Temp (°C) By Time (UTC) 2023')
plt.xlabel('Time (UTC)')
plt.ylabel('Dew Point Temp (°C)')
plt.show()

#Monthly averages for temp
monthly_trend = year2019.groupby('Month')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'brown')
plt.title('Average Temp (°C) Per Month 2019')
plt.xlabel('Month')
plt.ylabel('Temp (°C)')
plt.show()



monthly_trend = year2020.groupby('Month')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'red')
plt.title('Average Temp (°C) Per Month 2020')
plt.xlabel('Month')
plt.ylabel('Temp (°C)')
plt.show()



monthly_trend = year2021.groupby('Month')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'green')
plt.title('Average Temp (°C) Per Month 2021')
plt.xlabel('Month')
plt.ylabel('Temp (°C)')
plt.show()



monthly_trend = year2022.groupby('Month')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'orange')
plt.title('Average Temp (°C) Per Month 2022')
plt.xlabel('Month')
plt.ylabel('Temp (°C)')
plt.show()



monthly_trend = year2023.groupby('Month')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
monthly_trend.plot(kind='bar', color = 'purple')
plt.title('Average Temp (°C) Per Month 2023')
plt.xlabel('Month')
plt.ylabel('Temp (°C)')
plt.show()



#daily averages for temp

daily_trend = year2019.groupby('Day')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'brown')
plt.title('Average Temp (°C) Per Day 2019')
plt.xlabel('Day')
plt.ylabel('Temp (°C)')
plt.show()

daily_trend = year2020.groupby('Day')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'red')
plt.title('Average Temp (°C) Per Day 2020')
plt.xlabel('Day')
plt.ylabel('Temp (°C)')
plt.show()

daily_trend = year2021.groupby('Day')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'green')
plt.title('Average Temp (°C) Per Day 2021')
plt.xlabel('Day')
plt.ylabel('Temp (°C)')
plt.show()

daily_trend = year2022.groupby('Day')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'orange')
plt.title('Average Temp (°C) Per Day 2022')
plt.xlabel('Day')
plt.ylabel('Temp (°C)')
plt.show()


daily_trend = year2023.groupby('Day')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
daily_trend.plot(kind='bar', color = 'purple')
plt.title('Average Temp (°C) Per Day 2023')
plt.xlabel('Day')
plt.ylabel('Temp (°C)')
plt.show()

#hourly averages for temp

hourly_trend = year2019.groupby('Time (UTC)')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'brown')
plt.title('Average Temp (°C) By Time (UTC) 2019')
plt.xlabel('Time (UTC)')
plt.ylabel('Temp (°C)')
plt.show()

hourly_trend = year2020.groupby('Time (UTC)')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'red')
plt.title('Average Temp (°C) By Time (UTC) 2020')
plt.xlabel('Time (UTC)')
plt.ylabel('Temp (°C)')
plt.show()

hourly_trend = year2021.groupby('Time (UTC)')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'green')
plt.title('Average Temp (°C) By Time (UTC) 2021')
plt.xlabel('Time (UTC)')
plt.ylabel('Temp (°C)')
plt.show()

hourly_trend = year2022.groupby('Time (UTC)')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'orange')
plt.title('Average Temp (°C) By Time (UTC) 2022')
plt.xlabel('Time (UTC)')
plt.ylabel('Temp (°C)')
plt.show()

hourly_trend = year2023.groupby('Time (UTC)')['Temp (°C)'].mean()
plt.figure(figsize=(8, 4))
hourly_trend.plot(kind='bar', color = 'purple')
plt.title('Average Temp (°C) By Time (UTC) 2023')
plt.xlabel('Time (UTC)')
plt.ylabel('Temp (°C)')
plt.show()

# time series plots for temp, dew point temp, wind chill

plt.figure(figsize=(12, 6))
plt.plot(data['Wind Chill'], linewidth=0.5, color = 'blue')
plt.title('Wind Chill Time Series')
plt.xlabel('Time')
plt.ylabel('Wind Chill')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data['Dew Point Temp (°C)'], linewidth=0.5, color = 'blue')
plt.title('Dew Point Temp (°C) Time Series')
plt.xlabel('Time')
plt.ylabel('Dew Point Temp (°C)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data['Temp (°C)'], linewidth=0.5, color = 'blue')
plt.title('Temp (°C) Time Series')
plt.xlabel('Time')
plt.ylabel('Temp (°C)')
plt.legend()
plt.show()




#checking stationarity
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['Temp (°C)'])
print('ADF Stat:', result[0])
print('p-value:', result[1])

if result[1] > 0.05:
    print('Series is non-stationary. Differencing required')
else:
    print('Series is stationary')

# viewing ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data['Temp (°C)'], lags = 50)

plot_pacf(data['Temp (°C)'])



from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(data['Temp (°C)'], model = 'additive', period = 24)
decomp.plot()

plt.xticks([
    pd.to_datetime('2019-01-01 00:00:00'),
    pd.to_datetime('2020-01-01 00:00:00'),
    pd.to_datetime('2021-01-01 00:00:00'),
    pd.to_datetime('2022-01-01 00:00:00'),
    pd.to_datetime('2023-01-01 00:00:00')])

plt.show()

#residuals
sns.histplot(decomp.resid)


#looking at portion of the data to see clearly
#not including december because plot loses clarity and december is missing alot of data, less than 10 days of dec included each year

year2019 = data.query('Year == 2019 and Month <=3')
print(year2019)


decomp = seasonal_decompose(year2019['Temp (°C)'], model = 'additive', period = 24)
decomp.plot()

plt.show()


year2020 = data.query('Year == 2020 and Month <=3')
print(year2020)


decomp = seasonal_decompose(year2020['Temp (°C)'], model = 'additive', period = 24)
decomp.plot()

plt.show()


year2021 = data.query('Year == 2021 and Month <=3')
print(year2021)


decomp = seasonal_decompose(year2021['Temp (°C)'], model = 'additive', period = 24)
decomp.plot()

plt.show()

year2022 = data.query('Year == 2022 and Month <=3')
print(year2019)


decomp = seasonal_decompose(year2022['Temp (°C)'], model = 'additive', period = 24)
decomp.plot()

plt.show()

year2023 = data.query('Year == 2023 and Month <=3')
print(year2023)


decomp = seasonal_decompose(year2023['Temp (°C)'], model = 'additive', period = 24)
decomp.plot()

plt.show()



#creating dataset for analysis using only highly correlated variables to temp

highcorr = ['Dew Point Temp (°C)', 'Wind Chill', 'Temp (°C)']
prac = data[highcorr]

print(prac)


# train and test split
train = prac.loc['2019-01-01':'2022-12-31']
test = prac.loc['2023-01-01':'2023-12-31']




print(train)
print(len(train))
print(len(test))
print(test)

print(len(data))


#Finding ARIMA parameters

from pmdarima import auto_arima
model1 = auto_arima(train['Temp (°C)'],
                          seasonal=False,          
                         trace=True,
                         error_action='ignore',    
                          suppress_warnings=True,  
                         stepwise=True, d = 0)         
model1.summary()

#Best model:  ARIMA(3,0,5)(0,0,0)[0] intercept
#AIC                          17957.587



#Finding SARIMAX parameters 
from pmdarima import auto_arima
model2 = auto_arima(train['Temp (°C)'],exog = train.drop(columns = ['Temp (°C)']),
                          seasonal=True,          
                         trace=True,
                         error_action='ignore',    
                          suppress_warnings=True,  
                         stepwise=True, D = 0, d = 0, m =24)         
model2.summary()

#Best model:  ARIMA(3,0,5)(0,0,0)[24] intercept
#AIC                          17957.587




# ARIMA modelling
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


model_arima = ARIMA(train['Temp (°C)'], order=(3, 0, 5))
arima_result = model_arima.fit()
arima_result.summary()

arimapred = arima_result.forecast(steps=len(test))

plt.plot(test.index, test['Temp (°C)'], label='Test', color='orange', linewidth=0.25)

plt.plot(test.index, arimapred, label='Predicted', color='green', linewidth=0.25)


plt.xlabel('Time Steps')
plt.ylabel('Temperature (°C)')
plt.title('ARIMA - Test, and Predicted Data')
plt.legend()

plt.show()

mse_arima = mean_squared_error(test['Temp (°C)'], arimapred) 

print('ARIMA MSE:', mse_arima)


mae_arima = mean_absolute_error(test['Temp (°C)'], arimapred)

print('MAE ARIMA:', mae_arima)







#SARIMAX modelling
from statsmodels.tsa.statespace.sarimax import SARIMAX


model_sarimax = SARIMAX(
    train['Temp (°C)'], 
    exog=train.drop(columns=['Temp (°C)']),
    order=(3, 0, 5),  
    seasonal_order=(0,0,0,24)  
)
sarimax_result = model_sarimax.fit()

sarimax_result.summary()

forecast_sarimax = sarimax_result.forecast(steps=len(test),
                                         exog=test[['Dew Point Temp (°C)', 'Wind Chill']])





plt.plot(test.index, test['Temp (°C)'], label='Test', color='orange', linewidth=0.25)
plt.plot(test.index, forecast_sarimax, label='Predicted', color='green', linewidth=0.25)


plt.xlabel('Time Steps')
plt.ylabel('Temperature (°C)')
plt.title('SARIMAX - Test, and Predicted Data')
plt.legend()
plt.show()

mse_sarimax = mean_squared_error(test['Temp (°C)'], forecast_sarimax) 

print('SARIMAX MSE:', mse_sarimax)


mae_sarimax = mean_absolute_error(test['Temp (°C)'], forecast_sarimax)

print('MAE SARIMAX:', mae_sarimax)









#LSTM modelling
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

sequence_length = 24
X_train, y_train = [], []
X_test, y_test = [], []


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Temp (°C)']])


scaled_train = scaled_data[data.index.isin(train.index)]
scaled_test = scaled_data[data.index.isin(test.index)]


for i in range(len(scaled_train) - sequence_length):
    X_train.append(scaled_train[i:i + sequence_length])
    y_train.append(scaled_train[i + sequence_length])


for i in range(len(scaled_test) - sequence_length):
    X_test.append(scaled_test[i:i + sequence_length])
    y_test.append(scaled_test[i + sequence_length])


X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)


model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(1)
])


model_lstm.compile(optimizer='adam', loss='mean_squared_error')


model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)


lstm_predictions = model_lstm.predict(X_test)


lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)
y_test_rescaled = scaler.inverse_transform(y_test)


adjusted_test_index = test.index[-len(y_test_rescaled):]  


plt.plot(adjusted_test_index, y_test_rescaled, label='Test', color='orange', linewidth = 0.25)


plt.plot(adjusted_test_index, lstm_predictions_rescaled, label='Predicted', color='green', linewidth = 0.25)


plt.title('LSTM Model - Test, and Predicted Data')
plt.xlabel('Time Steps')
plt.ylabel('Temperature (°C)')
plt.legend()


plt.show()

mse_lstm = mean_squared_error(y_test_rescaled, lstm_predictions_rescaled)
print('LSTM MSE:',mse_lstm)


mae_lstm = mean_absolute_error(y_test_rescaled, lstm_predictions_rescaled)

print('MAE LSTM:', mae_lstm)




