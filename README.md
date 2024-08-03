# Uber-Demand-and-Supply-Analysis

### Uber demand supply Analysis 2024 practice



```python

```

- Importing Libraries 


```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
from datetime import datetime
now = datetime.now()
```

#### Task 1 :- Identify the data quality issues and clean the data so that you can use it for analysis.


```python
#Loading Dataset , dataset is in csv format.
uber = pd.read_csv('Uber Request Data.csv') 
uber.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Request id</th>
      <th>Pickup point</th>
      <th>Driver id</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619</td>
      <td>Airport</td>
      <td>1.0</td>
      <td>Trip Completed</td>
      <td>11/7/2016 11:51</td>
      <td>11/7/2016 13:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>867</td>
      <td>Airport</td>
      <td>1.0</td>
      <td>Trip Completed</td>
      <td>11/7/2016 17:57</td>
      <td>11/7/2016 18:47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1807</td>
      <td>City</td>
      <td>1.0</td>
      <td>Trip Completed</td>
      <td>12/7/2016 9:17</td>
      <td>12/7/2016 9:58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2532</td>
      <td>Airport</td>
      <td>1.0</td>
      <td>Trip Completed</td>
      <td>12/7/2016 21:08</td>
      <td>12/7/2016 22:03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3112</td>
      <td>City</td>
      <td>1.0</td>
      <td>Trip Completed</td>
      <td>13-07-2016 08:33:16</td>
      <td>13-07-2016 09:25:47</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3879</td>
      <td>Airport</td>
      <td>1.0</td>
      <td>Trip Completed</td>
      <td>13-07-2016 21:57:28</td>
      <td>13-07-2016 22:28:59</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4270</td>
      <td>Airport</td>
      <td>1.0</td>
      <td>Trip Completed</td>
      <td>14-07-2016 06:15:32</td>
      <td>14-07-2016 07:13:15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5510</td>
      <td>Airport</td>
      <td>1.0</td>
      <td>Trip Completed</td>
      <td>15-07-2016 05:11:52</td>
      <td>15-07-2016 06:07:52</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6248</td>
      <td>City</td>
      <td>1.0</td>
      <td>Trip Completed</td>
      <td>15-07-2016 17:57:27</td>
      <td>15-07-2016 18:50:51</td>
    </tr>
    <tr>
      <th>9</th>
      <td>267</td>
      <td>City</td>
      <td>2.0</td>
      <td>Trip Completed</td>
      <td>11/7/2016 6:46</td>
      <td>11/7/2016 7:25</td>
    </tr>
  </tbody>
</table>
</div>




```python
uber.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6745 entries, 0 to 6744
    Data columns (total 6 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Request id         6745 non-null   int64  
     1   Pickup point       6745 non-null   object 
     2   Driver id          4095 non-null   float64
     3   Status             6745 non-null   object 
     4   Request timestamp  6745 non-null   object 
     5   Drop timestamp     2831 non-null   object 
    dtypes: float64(1), int64(1), object(4)
    memory usage: 316.3+ KB
    


```python
uber.isnull().sum()
```




    Request id              0
    Pickup point            0
    Driver id            2650
    Status                  0
    Request timestamp       0
    Drop timestamp       3914
    dtype: int64




```python
uber['Driver id'].isnull().value_counts()
```




    Driver id
    False    4095
    True     2650
    Name: count, dtype: int64




```python
uber['Drop timestamp'].isnull().value_counts()
```




    Drop timestamp
    True     3914
    False    2831
    Name: count, dtype: int64




```python
uber=uber.drop(['Driver id'], axis=1)
uber=uber.drop(['Request id'], axis=1)
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>11/7/2016 11:51</td>
      <td>11/7/2016 13:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>11/7/2016 17:57</td>
      <td>11/7/2016 18:47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>12/7/2016 9:17</td>
      <td>12/7/2016 9:58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>12/7/2016 21:08</td>
      <td>12/7/2016 22:03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>13-07-2016 08:33:16</td>
      <td>13-07-2016 09:25:47</td>
    </tr>
  </tbody>
</table>
</div>



 - as we can see that Request timestamp which is related to timeseries data and having diffrent format , so we need to make it in dd mm yyyy format .


```python
uber['Request timestamp'] = pd.to_datetime(uber['Request timestamp'], dayfirst = True,format='mixed')
uber['Drop timestamp'] = pd.to_datetime(uber['Drop timestamp'], dayfirst = True,format='mixed')
# as per the updated pd.to_datetime() previously we didn't need to apply any sort of hyperparameters , 
#but now we required dayfirst = True,format='mixed'
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
    </tr>
  </tbody>
</table>
</div>




```python
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
    </tr>
  </tbody>
</table>
</div>



#### Task 1.2 Checking for Status col unique values with their particular sum


```python
uber['Status'].unique()
```




    array(['Trip Completed', 'Cancelled', 'No Cars Available'], dtype=object)




```python
status_summary = uber.groupby('Status').count()
status_summary.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 3 entries, Cancelled to Trip Completed
    Data columns (total 3 columns):
     #   Column             Non-Null Count  Dtype
    ---  ------             --------------  -----
     0   Pickup point       3 non-null      int64
     1   Request timestamp  3 non-null      int64
     2   Drop timestamp     3 non-null      int64
    dtypes: int64(3)
    memory usage: 96.0+ bytes
    


```python
status_summary.plot.bar(colormap = "winter")
plt.show()
```


    
![png](output_17_0.png)
    



```python
uber['Status'].count()
```




    6745



#### Task 1/07/2024 :- plotting graphs and binning time slots into particular hours of the day

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23  
1-6 early morning
6-12 morning
12-17 afternoon 
17-20 evening 
20-24 night  
step 1 :- check rows of req_timestmp 
step 2 :- if time matches particular range of slot 



```python

```


```python
def hr_func(ts):
    return ts.hour

uber['req_hour'] = uber['Request timestamp'].apply(hr_func)

```


```python
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
uber.loc[(uber['req_hour'] <= 6) & (uber['req_hour'] > 0 ),'req_timeslot']  = 'Early morning'
uber.loc[(uber['req_hour'] <=12) & (uber['req_hour'] > 6 ),'req_timeslot']  = 'Morning'
uber.loc[(uber['req_hour'] <=17) & (uber['req_hour'] > 12 ),'req_timeslot']  = 'Afternoon'
uber.loc[(uber['req_hour'] <=20) & (uber['req_hour'] > 17 ),'req_timeslot']  = 'Evening'
uber.loc[(uber['req_hour'] <=24) & (uber['req_hour'] > 20 ),'req_timeslot']  = 'Night'




#dividing the trips into 6 sessions based on dt.hour from Request Timestamp

# time_slots=['Late Night','Early Morning','Late Morning','Afternoon','Evening','Night']
# df_uber=df_uber.assign(time_slots=pd.cut(df_uber.Request_timestamp.dt.hour,[-1,4,8,12,16,20,24],labels=time_slots))
# df_uber.head()

```


```python
uber['req_hour'],uber['req_timeslot'].head(10)

```




    (0       11
     1       17
     2        9
     3       21
     4        8
             ..
     6740    23
     6741    23
     6742    23
     6743    23
     6744    23
     Name: req_hour, Length: 6745, dtype: int64,
     0          Morning
     1        Afternoon
     2          Morning
     3            Night
     4          Morning
     5            Night
     6    Early morning
     7    Early morning
     8        Afternoon
     9    Early morning
     Name: req_timeslot, dtype: object)




```python
uber['req_timeslot'].unique()
```




    array(['Morning', 'Afternoon', 'Night', 'Early morning', 'Evening', nan],
          dtype=object)




```python
uber['req_hour'].unique()
```




    array([11, 17,  9, 21,  8,  6,  5, 12,  4, 14, 22, 10, 18, 15,  2, 13, 16,
           19,  7, 20,  0,  3, 23,  1], dtype=int64)




```python
uber['req_timeslot'].isna()
```




    0       False
    1       False
    2       False
    3       False
    4       False
            ...  
    6740    False
    6741    False
    6742    False
    6743    False
    6744    False
    Name: req_timeslot, Length: 6745, dtype: bool




```python
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
      <td>11</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
      <td>17</td>
      <td>Afternoon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
      <td>9</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
      <td>21</td>
      <td>Night</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
      <td>8</td>
      <td>Morning</td>
    </tr>
  </tbody>
</table>
</div>




```python
uber_bkp=uber.copy()
```


```python
uber['req_hour']
```




    0       11
    1       17
    2        9
    3       21
    4        8
            ..
    6740    23
    6741    23
    6742    23
    6743    23
    6744    23
    Name: req_hour, Length: 6745, dtype: int64




```python
# part_of_the_day(frequency and availability of cabs from both locations)
# day_of_the_week(check for weekday weekend rush )
# cabs avg duration for trips completed ----- if time permits 

```

- Suggestion req


```python
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
      <td>11</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
      <td>17</td>
      <td>Afternoon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
      <td>9</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
      <td>21</td>
      <td>Night</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
      <td>8</td>
      <td>Morning</td>
    </tr>
  </tbody>
</table>
</div>




```python
# air-city 
# city-air 
# ----- air to city , time duration graph of count no of trips


# time series data 
# axis (time , range/freq)

```


```python
# colors = ["#CC2529", "#8E8D8D","#008000"]
plt.style.use('ggplot')
uber.groupby(['Pickup point','Status']).size().unstack().plot.bar(colormap = "winter",legend = True,figsize = (5,5))
plt.xlabel('Pickup point')
plt.ylabel('Total Count of Trip Status')
plt.show()
```


    
![png](output_36_0.png)
    


- 1. Cancelled :- City cancellations more than airport 
- 2. No Cars available :- City has more cars availble than airport (city is more populated region)
- 3. Trip Completed :- More trips completed from city than airport 


```python
uber.groupby(['Pickup point','Status']).size()
```




    Pickup point  Status           
    Airport       Cancelled             198
                  No Cars Available    1713
                  Trip Completed       1327
    City          Cancelled            1066
                  No Cars Available     937
                  Trip Completed       1504
    dtype: int64



- 1. Cancelled :- City cancellations more than airport
    1.1:- City cancellations more happen in the early-Mornings, and Mornings
    1.2:- Cancellations in the evening nad nigh is leeser for both city and airport 
- 2. No Cars available :- City has more cars availble than airport (city is more populated region)
    2.1:- No cars available during evening and night for pickups from airport
    2.2:- City has fewers cars available during mornings and early mornings for airport drop(more cars may available for city runs)
- 3. Trip Completed :- More trips completed from city than airport 
    3.1:- in  mornings 
    3.2:- During nights trip completed are fewers in both cases.
    


```python
# plotting frequency of all "Trip Status" over the hour of day 
plt.style.use('ggplot')
# colors = ['cool']
uber.groupby(['req_timeslot','Status']).Status.count().unstack().plot.bar(legend=True, figsize=(6,5), colormap="winter")
plt.title('Total Count of all Trip Statuses')
plt.xlabel('time_slots')
plt.ylabel('Total Count of Trip Status')
plt.show()
```


    
![png](output_40_0.png)
    



```python
# Filtering out only "Cancelled"  trips
df_tripscancelled=uber[uber["Status"].str.contains('Cancelled')==True]
df_tripscancelled=df_tripscancelled.reset_index(drop=True)
df_tripscancelled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City</td>
      <td>Cancelled</td>
      <td>2016-07-13 06:08:41</td>
      <td>NaT</td>
      <td>6</td>
      <td>Early morning</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City</td>
      <td>Cancelled</td>
      <td>2016-07-14 17:07:58</td>
      <td>NaT</td>
      <td>17</td>
      <td>Afternoon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Airport</td>
      <td>Cancelled</td>
      <td>2016-07-14 20:51:37</td>
      <td>NaT</td>
      <td>20</td>
      <td>Evening</td>
    </tr>
    <tr>
      <th>3</th>
      <td>City</td>
      <td>Cancelled</td>
      <td>2016-07-15 10:12:40</td>
      <td>NaT</td>
      <td>10</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Airport</td>
      <td>Cancelled</td>
      <td>2016-07-12 19:14:00</td>
      <td>NaT</td>
      <td>19</td>
      <td>Evening</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filtering out only "Trip Completed"  trips
df_tripscompleted=uber[uber["Status"].str.contains('Trip Completed')==True]
df_tripscompleted=df_tripscompleted.reset_index(drop=True)
df_tripscompleted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
      <td>11</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
      <td>17</td>
      <td>Afternoon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
      <td>9</td>
      <td>Morning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
      <td>21</td>
      <td>Night</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
      <td>8</td>
      <td>Morning</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filtering out only "No Cars Available"  trips
df_nocars=uber[uber["Status"].str.contains('No Cars Available')==True]
df_nocars=df_nocars.reset_index(drop=True)
df_nocars.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City</td>
      <td>No Cars Available</td>
      <td>2016-07-11 00:02:00</td>
      <td>NaT</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>City</td>
      <td>No Cars Available</td>
      <td>2016-07-11 00:06:00</td>
      <td>NaT</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>No Cars Available</td>
      <td>2016-07-11 00:09:00</td>
      <td>NaT</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>No Cars Available</td>
      <td>2016-07-11 00:23:00</td>
      <td>NaT</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Airport</td>
      <td>No Cars Available</td>
      <td>2016-07-11 00:30:00</td>
      <td>NaT</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



- sns countplot
<!-- # # Grouping by Status and Pickup point. -->
<!-- # uber.groupby(['Status','Pickup point']).size() -->
<!-- # # Visualize Status and Pickup point -->
<!-- # sns.countplot(x=uber['Pickup point'] , hue = uber['Status'] ,data = uber) -->


```python

```


```python
# plotting share/frequency of all Cancelled trips over the day to identify problem areas
plt.style.use('ggplot')
df_tripscompleted.groupby(['req_timeslot','Pickup point']).Status.count().unstack().plot.bar(legend=True, figsize=(6,6), colormap="winter")
plt.title('Count and Distribution of all "Cancelled" Trips over the day')
plt.xlabel('req_timeslot')
plt.ylabel('Total Count of "Completed" Trips')
plt.show()
```


    
![png](output_46_0.png)
    



```python
# plotting share/frequency of all Cancelled trips over the day to identify problem areas
plt.style.use('ggplot')
df_nocars.groupby(['req_timeslot','Pickup point']).Status.count().unstack().plot.bar(legend=True, figsize=(6,6), colormap="winter")
plt.title('Count and Distribution of all "Cancelled" Trips over the day')
plt.xlabel('req_timeslot')
plt.ylabel('Total Count of "No Cars available" Trips')
plt.show()
```


    
![png](output_47_0.png)
    



```python
uber['supply_demand'] = ['Supply' if x == 'Trip Completed' else 'Demand' for x in uber['Status']]
uber.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
      <th>supply_demand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6735</th>
      <td>Airport</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:39:15</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
    <tr>
      <th>6736</th>
      <td>Airport</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:42:51</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
    <tr>
      <th>6737</th>
      <td>City</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:43:54</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
    <tr>
      <th>6738</th>
      <td>City</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:46:03</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
    <tr>
      <th>6739</th>
      <td>City</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:46:20</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
    <tr>
      <th>6740</th>
      <td>City</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:49:03</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
    <tr>
      <th>6741</th>
      <td>Airport</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:50:05</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
    <tr>
      <th>6742</th>
      <td>City</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:52:06</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
    <tr>
      <th>6743</th>
      <td>City</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:54:39</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
    <tr>
      <th>6744</th>
      <td>Airport</td>
      <td>No Cars Available</td>
      <td>2016-07-15 23:55:03</td>
      <td>NaT</td>
      <td>23</td>
      <td>Night</td>
      <td>Demand</td>
    </tr>
  </tbody>
</table>
</div>




```python
uber['supply_demand'].value_counts()
```




    supply_demand
    Demand    3914
    Supply    2831
    Name: count, dtype: int64




```python
#Plotting Supply and Demand on the City to Airport Route
df_citytoairport_supplydemand=uber[uber['Pickup point'].str.contains('City')==True]
plt.style.use('ggplot')
df_citytoairport_supplydemand.groupby(['req_timeslot','supply_demand']).supply_demand.count().unstack().plot.bar(legend=True, figsize=(5,5))
plt.title('Supply-Demand curve for City to Airport Route')
plt.xlabel('time_slots')
plt.ylabel('Supply/Demand')
plt.show()
```


    
![png](output_50_0.png)
    


- Cancelled :- City cancellations more than airport 1.1:- City cancellations more happen in the early-Mornings, and Mornings 1.2:- Cancellations in the evening nad nigh is leeser for both city and airport
No Cars available :- City has more cars availble than airport (city is more populated region) 2.1:- No cars available during evening and night for pickups from airport 2.2:- City has fewers cars available during mornings and early mornings for airport drop(more cars may available for city runs)
Trip Completed :- More trips completed from city than airport 3.1:- in mornings 3.2:- During nights trip completed are fewers in both cases.m


```python
#Plotting Supply and Demand on the Airport to City route
df_airporttocity_supplydemand=uber[uber['Pickup point'].str.contains('Airport')==True]
plt.style.use('ggplot')
df_airporttocity_supplydemand.groupby(['req_timeslot','supply_demand']).supply_demand.count().unstack().plot.bar(legend=True, figsize=(5,5))
plt.title('Supply-Demand curve for Airport to City Route')
plt.xlabel('req_timeslot')
plt.ylabel('Supply/Demand')
plt.show()
```


    
![png](output_52_0.png)
    


- Date ,Day ,Year` function task 


```python
def hr_func(ts):
    return ts.day

uber['req_day'] = uber['Request timestamp'].apply(hr_func)

```


```python
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
      <th>supply_demand</th>
      <th>req_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
      <td>11</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
      <td>17</td>
      <td>Afternoon</td>
      <td>Supply</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
      <td>9</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
      <td>21</td>
      <td>Night</td>
      <td>Supply</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
      <td>8</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
def hr_func(ts):
    return ts.month

uber['req_month'] = uber['Request timestamp'].apply(hr_func)

```


```python
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
      <th>supply_demand</th>
      <th>req_day</th>
      <th>req_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
      <td>11</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
      <td>17</td>
      <td>Afternoon</td>
      <td>Supply</td>
      <td>11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
      <td>9</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>12</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
      <td>21</td>
      <td>Night</td>
      <td>Supply</td>
      <td>12</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
      <td>8</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>13</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
def hr_func(ts):
    return ts.year

uber['req_year'] = uber['Request timestamp'].apply(hr_func)

```


```python
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
      <th>supply_demand</th>
      <th>req_day</th>
      <th>req_month</th>
      <th>req_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
      <td>11</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>11</td>
      <td>7</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
      <td>17</td>
      <td>Afternoon</td>
      <td>Supply</td>
      <td>11</td>
      <td>7</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
      <td>9</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>12</td>
      <td>7</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
      <td>21</td>
      <td>Night</td>
      <td>Supply</td>
      <td>12</td>
      <td>7</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
      <td>8</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>13</td>
      <td>7</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
</div>




```python
def hr_func(ts):
    return ts.min

uber['req_min'] = uber['Request timestamp'].apply(hr_func)

```


```python
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
      <th>supply_demand</th>
      <th>req_day</th>
      <th>req_month</th>
      <th>req_year</th>
      <th>req_min</th>
      <th>weekdays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
      <td>11</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>11</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
      <td>17</td>
      <td>Afternoon</td>
      <td>Supply</td>
      <td>11</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
      <td>9</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>12</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
      <td>21</td>
      <td>Night</td>
      <td>Supply</td>
      <td>12</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
      <td>8</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>13</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# uber['Day'] = uber['Request timestamp'].apply(lambda x: datetime.datetime.strftime(x, '%A'))
```


```python
def hr_func(ts):
    return ts.strftime('%A')

uber['weekdays'] = uber['Request timestamp'].apply(hr_func)

```


```python
uber.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pickup point</th>
      <th>Status</th>
      <th>Request timestamp</th>
      <th>Drop timestamp</th>
      <th>req_hour</th>
      <th>req_timeslot</th>
      <th>supply_demand</th>
      <th>req_day</th>
      <th>req_month</th>
      <th>req_year</th>
      <th>req_min</th>
      <th>weekdays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 11:51:00</td>
      <td>2016-07-11 13:00:00</td>
      <td>11</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>11</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-11 17:57:00</td>
      <td>2016-07-11 18:47:00</td>
      <td>17</td>
      <td>Afternoon</td>
      <td>Supply</td>
      <td>11</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-12 09:17:00</td>
      <td>2016-07-12 09:58:00</td>
      <td>9</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>12</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Airport</td>
      <td>Trip Completed</td>
      <td>2016-07-12 21:08:00</td>
      <td>2016-07-12 22:03:00</td>
      <td>21</td>
      <td>Night</td>
      <td>Supply</td>
      <td>12</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>City</td>
      <td>Trip Completed</td>
      <td>2016-07-13 08:33:16</td>
      <td>2016-07-13 09:25:47</td>
      <td>8</td>
      <td>Morning</td>
      <td>Supply</td>
      <td>13</td>
      <td>7</td>
      <td>2016</td>
      <td>1677-09-21 00:12:43.145224193</td>
      <td>Wednesday</td>
    </tr>
  </tbody>
</table>
</div>




```python
from datetime import datetime
now = datetime.now()
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted)
```

    2024-07-11 18:16:01
    


```python

```
