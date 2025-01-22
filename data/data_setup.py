import pandas as pd
from datetime import datetime
import kagglehub


def mergedata():
    """
    This function downloads the dataset from Kaggle and merges the two datasets
    based on location and datetime columns.
    """
    # Download latest version
    path = kagglehub.dataset_download("ravi72munde/uber-lyft-cab-prices")

    print("Path to dataset files:", path)

    df = pd.read_csv(path + '/cab_rides.csv')
    weather = pd.read_csv(path + '/weather.csv')

    # merging 2 datasets base on location and datatime
    weather['time_stamp'] = [datetime.fromtimestamp(i) for
                             i in weather['time_stamp']]
    weather['time_stamp'] = weather['time_stamp'].values.astype(
        'datetime64[h]')
    df['time_stamp'] = [datetime.fromtimestamp(i/1000.0) for
                        i in df['time_stamp']]
    df['time_stamp'] = df['time_stamp'].values.astype('datetime64[h]')
    df.rename(columns={'source': 'location'},
              inplace=True)
    weather.drop_duplicates(['time_stamp', 'location'],
                            inplace=True)

    merged_df = pd.merge(df,
                         weather,
                         on=['location', 'time_stamp'])

    # add 2 new feature base on time_stamp column
    merged_df['hour'] = merged_df['time_stamp'].dt.hour
    merged_df['day'] = merged_df['time_stamp'].dt.day_name()

    return merged_df


def splitdata(data):
    """
    This function splits the data into training and testing datasets.

    Parameters
    ----------
    data : DataFrame
        The dataset to be split.

    Returns
    -------
    train_data : DataFrame
        The training dataset. 70% of the data.
    valid_data : DataFrame
        The validation dataset. 15% of the data.
    test_data : DataFrame
        The testing dataset. 15% of the data.
    """
    # split data into training and testing datasets
    train_data = data.sample(frac=0.7, random_state=0)
    valid_data = data.drop(train_data.index).sample(frac=0.5, random_state=0)
    test_data = data.drop(train_data.index.drop(valid_data.index))

    return train_data, test_data, valid_data
