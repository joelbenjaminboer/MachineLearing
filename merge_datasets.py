import pandas as pd
from datetime import datetime

df = pd.read_csv('Datasets/cab_rides.csv')
weather = pd.read_csv('Datasets/weather.csv')

# merging 2 datasets base on location and datatime
weather['time_stamp'] = [datetime.fromtimestamp(i) for
                         i in weather['time_stamp']]
weather['time_stamp'] = weather['time_stamp'].values.astype('datetime64[h]')
df['time_stamp'] = [datetime.fromtimestamp(i/1000.0) for
                    i in df['time_stamp']]
df['time_stamp'] = df['time_stamp'].values.astype('datetime64[h]')
df.rename(columns={'source': 'location'},
          inplace=True)
weather.drop_duplicates(['time_stamp', 'location'],
                        inplace=True)

new_df = pd.merge(df,
                  weather,
                  on=['location', 'time_stamp'])

# add 2 new feature base on time_stamp column
new_df['hour'] = new_df['time_stamp'].dt.hour
new_df['day'] = new_df['time_stamp'].dt.day_name()
new_df.head(2)
