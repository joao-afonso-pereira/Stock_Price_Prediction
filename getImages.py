import os
import numpy as np
import glob
import plotly.graph_objects as go
import datetime
import pandas as pd
pd.options.mode.chained_assignment = None
  

def getData(filename):
  df = pd.read_csv(filename, index_col='Date',
                parse_dates=True, na_values=['nan'])
  return df


def candleStick(df, date_range):
  new_df = df.loc[date_range[0]:date_range[len(date_range)-1],:]
  fig = go.Figure(data=[go.Candlestick(x=new_df.index,
                  open=new_df['Open'], high=new_df['High'],
                  low=new_df['Low'], close=new_df['Close'])
                      ])

  fig.update_layout(xaxis_rangeslider_visible=False)
  #fig.show()
  
def whiteCandleStick(df, filename, path):
  layout =  go.Layout(
      xaxis = dict(
        showticklabels = False,
      ),
      yaxis = dict(
        showticklabels = False,
      ),
  )
  fig = go.Figure(data=[go.Candlestick(x=df.index,
                  open=df['Open'], high=df['High'],
                  low=df['Low'], close=df['Close'])
                      ], layout = layout)
  fig.update_layout(xaxis_rangeslider_visible=False, plot_bgcolor='rgb(255,255,255)')
  #fig.show()
  start_str = '{:%m_%d_%Y}'.format(df.index[0])
  end_str = '{:%m_%d_%Y}'.format(df.index[len(df)-1])
  name = filename + "_from_" + start_str + "_to_" + end_str
  fig.write_image("{}{}.png".format(path, name))


################################################################
################################################################
################################################################

print(datetime.datetime.now())
print("")
  
list_names = []

for filename in glob.glob("*.txt"):

  list_names.append(filename)

for i in range (304, len(list_names)): #len(list_names)
    
    filename = list_names[i].rstrip()
    
    empty = os.stat(filename).st_size == 0
    
    if empty == False:
    
        print("{} of {} - {}".format(i+1, len(list_names), filename))
        
        df = getData(filename)
        dates = df.index
        num = len(dates)
        num_of_years = len(dates)/260
        batch_size = 88
        
        #TRAIN
        df_train = df.loc[dates[0]:dates[int(num*0.9)], :]
        df_train=df_train.dropna()
        dates_train = df_train.index
        num_of_years_train = len(dates_train)/260
        
        num_of_days_train = batch_size
        num_of_occ_train = int(len(dates_train)/batch_size)
        
        
        #TEST
        df_test = df.loc[dates[int(num*0.9)]:dates[num-1], :]
        df_test=df_test.dropna()
        dates_test = df_test.index
        num_of_years_test = len(dates_test)/260
        
        num_of_days_test = batch_size
        num_of_occ_test = int(len(dates_test)/batch_size)
        
        
        ################################################################
        ################################################################
        ################################################################
        
        #TRAIN
        path = "../train_images/"
        
        train_label = []
        
        for i in range(0, num_of_occ_train):
        
          dates = df_train.index.tolist()
        
          start = i*num_of_days_train
          end = i*num_of_days_train + num_of_days_train - 1
        
          data = df_train.loc[dates[start]:dates[end], :]
        
          size = len(data)
          previous_data = data[:int(size*0.75)]
          next_month = data['Close'][size-1]
        
          #whiteCandleStick(previous_data, filename, path)
        
          if next_month > previous_data['Close'][len(previous_data)-1]:
            train_label.append(1)
          else:
            train_label.append(0)
        
        np.savetxt("../train_label.txt", np.array(train_label), fmt="%i")
        
        
        #TEST
        path = "../test_images/"
        
        test_label = []
        
        for i in range(0, num_of_occ_test):
        
          dates = df_test.index.tolist()
        
          start = i*num_of_days_test
          end = i*num_of_days_test + num_of_days_test - 1
        
          data = df_test.loc[dates[start]:dates[end], :]
        
          size = len(data)
          previous_data = data[:int(size*0.75)]
          next_month = data['Close'][size-1]
        
          #whiteCandleStick(previous_data, filename, path)
        
          if next_month > previous_data['Close'][len(previous_data)-1]:
            test_label.append(1)
          else:
            test_label.append(0)
        
        np.savetxt("../test_label.txt", np.array(test_label), fmt="%i")

print("")            
print(datetime.datetime.now())