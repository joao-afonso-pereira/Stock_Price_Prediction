import numpy as np
import os
import glob
import datetime
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.linear_model import LinearRegression
import calendar
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statistics import mean 

  
def getData(filename):
  df = pd.read_csv(filename, index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
  return df

def getDataRangeDates(filename, dates):
  df = pd.DataFrame(index=dates)
  df = df.join(pd.read_csv(filename, index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan']))
  return df


def check_if_last_day_of_month(date):
  last_day_of_month = calendar.monthrange(date.year, date.month)[1]
  if date == datetime.date(date.year, date.month, last_day_of_month):
    return True
  else:
    return False

print(datetime.datetime.now())
print("")
  
list_names = []

for filename in glob.glob("*.txt"):

  list_names.append(filename)

mae_lr = []
mae_svm = []
mae_knn = []
mae_rf = []

rmse_lr = []
rmse_svm = []
rmse_knn = []
rmse_rf = []

r2_lr = []
r2_svm = []
r2_knn = []
r2_rf = []

n_files = 1

print("Number of files per iteration: {}".format(n_files))

it = 1

for iteration in range(it):
    
    print("\n***Iteration {} of {}***".format(iteration+1, it))
    
    names = random.sample(list_names, k=n_files)

    for f in range(len(names)):
        
      filename = names[f]
      
      #if f % 50 == 0:
      print("\nFile {} of {}: {}".format(f+1, len(names), filename))
        
      filename = names[f]
        
      empty = os.stat(filename).st_size == 0
      
      df = getData(filename)
      df = df.dropna()
      dates = df.index.tolist()
        
      if empty == False and dates[0]<datetime.datetime(2009, 1, 1):
        
        df = getDataRangeDates(filename, pd.date_range('2010-01-01',dates[len(dates)-1],freq='B'))
        dates = df.index.tolist()
        
        num_of_years = len(dates)/260
        
        num_of_days = 260
        num_of_occ = int(num_of_years)
        
        data_list  = []
        y_true = []
        y_pred_lr = []
        y_pred_svm = []
        y_pred_knn = []
        y_pred_rf = []

        for i in range(0, num_of_occ):
            
            #print("\n*Batch {} of {}*".format(i+1, num_of_occ))
            
            dates = df.index.tolist()

            data = df.loc[dates[i*num_of_days]:dates[num_of_days*(i+1)-1], :]
            data_list.append(data)
        
            #Feature extraction
            dates = data.index.tolist()
        
            data.loc[:, 'year'] = 0
            data.loc[:, 'month'] = 0
            data.loc[:, 'day'] = 0
            data.loc[:, 'year_day'] = 0
            data.loc[:, 'month_start'] = 0
            data.loc[:, 'month_end'] = 0
            data.loc[:, 'week_day'] = 0
            data.loc[:, 'monday_or_friday'] = 0
        
            for i in range(0,len(dates)):
                data.loc[dates[i], 'year'] = dates[i].year
                data.loc[dates[i], 'month'] = dates[i].month
                data.loc[dates[i], 'day'] = dates[i].day
                
                data.loc[dates[i], 'year_day'] = dates[i].timetuple().tm_yday
            
                if(dates[i].day == 1):
                    data.loc[dates[i], 'month_start'] = 1
                else:
                    data.loc[dates[i], 'month_start'] = 0
            
                if(check_if_last_day_of_month(dates[i]) == True):
                    data.loc[dates[i], 'month_end'] = 1
                else:
                    data.loc[dates[i], 'month_end'] = 0
            
                data.loc[dates[i], 'week_day'] = dates[i].weekday()
                if (dates[i].weekday() == 0 or dates[i].weekday() == 4):
                  data.loc[dates[i], 'monday_or_friday'] = 1
                else:
                  data.loc[dates[i], 'monday_or_friday'] = 0   
                  
            #Data preparation
            size = len(data)
            train = data[:int(size*0.7)]
            train = train.dropna()
            val = data[int(size*0.7):]
            val = val.dropna()
            
            x_train = train.drop('Close', axis=1)
            y_train = train['Close']
            x_val = val.drop('Close', axis=1)
            y_val = val['Close']
            
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            y_val = np.array(y_val)
        
            y_val = np.reshape(y_val, (y_val.shape[0], 1))
            y_true.append(scaler.fit_transform(y_val))
            
            
            #Linear regression ---------------------------------------------------------------------------------------------------
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            preds = lr.predict(x_val)
            
            preds = np.array(preds)
            preds = np.reshape(preds, (preds.shape[0], 1))
            
            y_pred_lr.append(scaler.fit_transform(preds))           
            #print("Linear Regression: check!")
            
            x_train_scaled = scaler.fit_transform(x_train)
            x_train = pd.DataFrame(x_train_scaled)
            x_val_scaled = scaler.fit_transform(x_val)
            x_val = pd.DataFrame(x_val_scaled)
            
            #SVM ---------------------------------------------------------------------------------------------------
            x_train, x_otim, y_train, y_otim = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

            c_list = range(1, 5)
            kernel_list = ['linear', 'rbf']

            m1_scores = []
            c_values = []
            k_values = []
            
            for _c in c_list:
                for _k in kernel_list:

                    m1=svm.SVR(kernel=_k, C=_c)
                    m1.fit(x_train, y_train)
                    m1_scores.append(m1.score(x_otim, y_otim))
                    c_values.append(_c)
                    k_values.append(_k)
    
            result1 = np.where(m1_scores == np.amin(m1_scores))
            index1 = result1[0][0]

            sv=svm.SVR(kernel=k_values[index1], C=c_values[index1])
            sv.fit(x_train, y_train)
            preds = sv.predict(x_val)
            
            preds = np.array(preds)
            preds = np.reshape(preds, (preds.shape[0], 1))
            
            y_pred_svm.append(scaler.fit_transform(preds))
            #print("SVM: check!")
    
    
            #KNN ---------------------------------------------------------------------------------------------------
            params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
            knn = neighbors.KNeighborsRegressor()
            model = GridSearchCV(knn, params, cv=5)
            model.fit(x_train,y_train)
            preds = model.predict(x_val)
            
            preds = np.array(preds)
            preds = np.reshape(preds, (preds.shape[0], 1))
            
            y_pred_knn.append(scaler.fit_transform(preds))
            #print("KNN: check!")
            
            
            
            #Random Forest ---------------------------------------------------------------------------------------------------
            n_estimators_list = [*range(1, 101, 1)] 
            scores_list = []
            
            for n in n_estimators_list:
                rf = RandomForestRegressor(random_state=0, n_estimators=n)
                rf.fit(x_train,y_train)
                scores_list.append(rf.score(x_train, y_train))
            
            best_score = max(scores_list)
            index = scores_list.index(best_score)
            best_n = n_estimators_list[index]
            
            rf = RandomForestRegressor(random_state=0, n_estimators=best_n)
            rf.fit(x_train,y_train)
            preds = rf.predict(x_val)
            
            preds = np.array(preds)
            preds = np.reshape(preds, (preds.shape[0], 1))
    
            y_pred_rf.append(scaler.fit_transform(preds))
            #print("Random Forest: check!")

        mae_list_lr = []
        mae_list_svm = []
        mae_list_knn = []
        mae_list_rf = []
        
        rmse_list_lr = []
        rmse_list_svm = []
        rmse_list_knn = []
        rmse_list_rf = []
        
        r2_list_lr = []
        r2_list_svm = []
        r2_list_knn = []
        r2_list_rf = []
        
        for i in range(len(y_true)):
        
          #Mean Absolute Error
          mae_list_lr.append(mean_absolute_error(y_true[i], y_pred_lr[i]))
          mae_list_svm.append(mean_absolute_error(y_true[i], y_pred_svm[i]))
          mae_list_knn.append(mean_absolute_error(y_true[i], y_pred_knn[i]))
          mae_list_rf.append(mean_absolute_error(y_true[i], y_pred_rf[i]))
        
          #Root Mean Square Error
          rmse_list_lr.append(np.sqrt(mean_squared_error(y_true[i], y_pred_lr[i])))
          rmse_list_svm.append(np.sqrt(mean_squared_error(y_true[i], y_pred_svm[i])))
          rmse_list_knn.append(np.sqrt(mean_squared_error(y_true[i], y_pred_knn[i])))
          rmse_list_rf.append(np.sqrt(mean_squared_error(y_true[i], y_pred_rf[i])))
        
          #Coefficient of Determination (R^2)
          r2_list_lr.append(r2_score(y_true[i], y_pred_lr[i]))
          r2_list_svm.append(r2_score(y_true[i], y_pred_svm[i]))
          r2_list_knn.append(r2_score(y_true[i], y_pred_knn[i]))
          r2_list_rf.append(r2_score(y_true[i], y_pred_rf[i]))
        
        mae_lr.append(mean(mae_list_lr))
        mae_svm.append(mean(mae_list_svm))
        mae_knn.append(mean(mae_list_knn))
        mae_rf.append(mean(mae_list_rf))
        
        rmse_lr.append(mean(rmse_list_lr))
        rmse_svm.append(mean(rmse_list_svm))
        rmse_knn.append(mean(rmse_list_knn))
        rmse_rf.append(mean(rmse_list_rf))
        
        r2_lr.append(mean(r2_list_lr))
        r2_svm.append(mean(r2_list_svm))
        r2_knn.append(mean(r2_list_knn))
        r2_rf.append(mean(r2_list_rf))

mae_lr = mean(mae_lr)
mae_svm = mean(mae_svm)
mae_knn = mean(mae_knn)
mae_rf = mean(mae_rf)

print("\nMAE:\nLR = {:1.3f}\nSVM = {:1.3f}\nKNN = {:1.3f}\nRF = {:1.3f}".format(mae_lr, mae_svm, mae_knn, mae_rf))

rmse_lr = mean(rmse_lr)
rmse_svm = mean(rmse_svm)
rmse_knn = mean(rmse_knn)
rmse_rf = mean(rmse_rf)

print("\nRMSE:\nLR = {:1.3f}\nSVM = {:1.3f}\nKNN = {:1.3f}\nRF = {:1.3f}".format(rmse_lr, rmse_svm, rmse_knn, rmse_rf))

r2_lr = mean(r2_lr)
r2_svm = mean(r2_svm)
r2_knn = mean(r2_knn)
r2_rf = mean(r2_rf)

print("\nR2:\nLR = {:1.3f}\nSVM = {:1.3f}\nKNN = {:1.3f}\nRF = {:1.3f}".format(r2_lr, r2_svm, r2_knn, r2_rf))

print("")            
print(datetime.datetime.now())

    
    
    
