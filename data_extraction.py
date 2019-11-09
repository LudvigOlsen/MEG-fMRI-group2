import numpy as np
import pandas as pd


##------------------------------------------------------------------##
## Data Extraction Utils
##------------------------------------------------------------------##

def split_by_time_points(x):
  split_by = np.asarray(range(x.shape[2] - 1)) + 1
  time_splits = np.split(x, split_by, axis=2)
  time_splits = [ts.reshape((x.shape[0], -1)) for ts in time_splits]
  return time_splits


def to_data_frame(x, id):
  df = pd.DataFrame.from_records(x)
  df.columns = ["S_" + str(i) for i in range(len(df.columns))]
  df["Trial"] = range(len(df))
  df["Time Point"] = id
  return df


def create_time_point_data_frames(x):
  time_splits = split_by_time_points(x)
  return [to_data_frame(ts, i) for i, ts in enumerate(time_splits)]
