import numpy as np
import os
import glob
import datetime
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import random
  
def getData(filename):
  df = pd.read_csv(filename, index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
  return df

print(datetime.datetime.now())
print("")
  
list_names = []

for filename in glob.glob("*.txt"):

  list_names.append(filename)

train_data = []
train_label = []
test_data = []
test_label = []
batch_size = 88

rf_precision = []
nb_precision = []
svm_precision = []
knn_precision = []

rf_recall = []
nb_recall = []
svm_recall = []
knn_recall = []

print("Number of files: {}".format(100))

for iteration in range(15):
    
    print("\nIteration {} of {}".format(iteration+1, 15))
    
    names = random.sample(list_names, k=100)

    for filename in names:
        
      empty = os.stat(filename).st_size == 0
        
      if empty == False:
    
          df = getData(filename)
        
          scaler = MinMaxScaler(feature_range=(0, 1))
          scaled_data = scaler.fit_transform(df)
        
          dates = df.index
          num = len(dates)
        
          #TRAIN
          df_train = df.loc[:dates[int(num*0.9)], :]
          df_train=df_train.dropna()
          dates_train = df_train.index
          num_of_years_train = len(dates_train)/260
        
          num_of_days_train = batch_size
          num_of_occ_train = int(len(dates_train)/batch_size)
        
          for i in range(0, num_of_occ_train):
        
            dates = df_train.index.tolist()
        
            start = i*num_of_days_train
            end = i*num_of_days_train + num_of_days_train - 1
        
            data = df_train.loc[dates[start]:dates[end], :]
        
            size = len(data)
            previous_data = data[:int(size*0.75)]
        
            train_data.append(np.array(previous_data).reshape((previous_data.shape[1], previous_data.shape[0])))
        
            next_month = data['Close'][size-1]
        
            if next_month > previous_data['Close'][len(previous_data)-1]:
              train_label.append(1)
            else:
              train_label.append(0)
        
          #TEST
          df_test = df.loc[dates[int(num*0.9)]:, :]
          df_test=df_test.dropna()
          dates_test = df_test.index
          num_of_years_test = len(dates_test)/260
        
          num_of_days_test = batch_size
          num_of_occ_test = int(len(dates_test)/batch_size)
        
          for i in range(0, num_of_occ_test):
        
            dates = df_test.index.tolist()
        
            start = i*num_of_days_test
            end = i*num_of_days_test + num_of_days_test - 1
        
            data = df_test.loc[dates[start]:dates[end], :]
        
            size = len(data)
            previous_data = data[:int(size*0.75)]
        
            test_data.append(np.array(previous_data).reshape((previous_data.shape[1], previous_data.shape[0])))
        
            next_month = data['Close'][size-1]
        
            if next_month > previous_data['Close'][len(previous_data)-1]:
              test_label.append(1)
            else:
              test_label.append(0)
        
    x_train = np.concatenate(train_data, axis=0)
    y_train = np.array(train_label)
    
    x_test = np.concatenate(test_data, axis=0)
    y_test = np.array(test_label)
    
    #print("Train size: {}\nTest size: {}".format(len(y_train), len(y_test)))
        
    
    ### RF ###
    
    n_estimators_list = [*range(1, 101, 1)] 
    scores_list = []
    
    #for n in n_estimators_list:
    #  rf = RandomForestClassifier(n_estimators=n)
    #  rf.fit(x_train,y_train)
    #  scores_list.append(rf.score(x_train, y_train))
    
    #best_score = max(scores_list)
    #index = scores_list.index(best_score)
    #best_n = n_estimators_list[index]
    #rf = RandomForestClassifier(n_estimators=best_n)
    
    rf = RandomForestClassifier(n_estimators=7)
    
    rf.fit(x_train,y_train)
    y_pred_rf = rf.predict(x_test)
    
    #print("\nRandom Forest done!")
    
    #n=0
    ### SVM ###
    #kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
    #c_list = range(1, 10)
    #otimization_svm = np.ones((len(kernel_list), len(c_list)))
    #for _kernel in list(enumerate(kernel_list)):
    #    for _c in list(enumerate(c_list)):
    #        n=n+1
    #        print(n)
    #        svm = SVC(C=_c[1], kernel=_kernel[1])
    #        scores = cross_val_score(svm, x_train, y_train, cv=5)
    #        otimization_svm[_kernel[0], _c[0]] = scores.mean()
    #
    #result = np.where(otimization_svm == np.amax(otimization_svm))
    #listOfCordinates = list(zip(result[0], result[1]))
    #best_kernel = kernel_list[listOfCordinates[0][0]]
    #best_c = c_list[listOfCordinates[0][1]]
    #svm = SVC(C=best_c, kernel=best_kernel, probability=True)
    
    svm = SVC(C=1, kernel='rbf')
    
    svm.fit(x_train, y_train)
    y_pred_svm = svm.predict(x_test)
    
    #print("SVM done!")
    
    ### NB ###
    nb = GaussianNB()
    scores = cross_val_score(nb, x_train, y_train, cv=5)
    nb.fit(x_train, y_train)
    y_pred_nb = nb.predict(x_test)
    
    #print("Naive Bayes done!")
    
    ### KNN ###
    #weights_list = ['uniform', 'distance']
    #k_list = range(1, 10)
    #otimization_knn = np.ones((len(k_list), len(weights_list)))
    #for _w in list(enumerate(weights_list)):
    #    for _k in list(enumerate(k_list)):
    #        knn = KNeighborsClassifier(n_neighbors=_k[1], weights=_w[1])
    #        scores = cross_val_score(knn, x_train, y_train, cv=5)
    #        otimization_knn[_k[0], _w[0]] = scores.mean()
    #
    #result = np.where(otimization_knn == np.amax(otimization_knn))
    #listOfCordinates = list(zip(result[0], result[1]))
    #best_k = k_list[listOfCordinates[0][0]]
    #best_weight = weights_list[listOfCordinates[0][1]]
    #knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)
    
    knn = KNeighborsClassifier(n_neighbors=7, weights='uniform')
    
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    
    #print("KNN done!")
    
    
    rf_precision.append(precision_score(y_test, y_pred_rf))
    svm_precision.append(precision_score(y_test, y_pred_svm))
    nb_precision.append(precision_score(y_test, y_pred_nb))
    knn_precision.append(precision_score(y_test, y_pred_knn))
    
    rf_recall.append(recall_score(y_test, y_pred_rf))
    svm_recall.append(recall_score(y_test, y_pred_svm))
    nb_recall.append(recall_score(y_test, y_pred_nb))
    knn_recall.append(recall_score(y_test, y_pred_knn))

from statistics import mean 

rf_p = mean(rf_precision)
svm_p = mean(svm_precision)
nb_p = mean(nb_precision)
knn_p = mean(knn_precision)

rf_r = mean(rf_recall)
svm_r = mean(svm_recall)
nb_r = mean(nb_recall)
knn_r = mean(knn_recall)

print("\nRF precision = {:1.3f}".format(rf_p))
print("SVM precision = {:1.3f}".format(svm_p))
print("NB precision = {:1.3f}".format(nb_p))
print("KNN precision = {:1.3f}".format(knn_p))

print("\nRF recall = {:1.3f}".format(rf_r))
print("SVM recall = {:1.3f}".format(svm_r))
print("NB recall = {:1.3f}".format(nb_r))
print("KNN recall = {:1.3f}".format(knn_r))

    
print("")            
print(datetime.datetime.now())