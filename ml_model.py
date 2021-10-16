import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz


def makeLabel(Y) :
  label = []
  for i in Y:
    if(i=='No') :
      label.append("0")
    else :
      label.append("1")
  return label
weather = pd.read_csv('/home/nivedha/weather-pred/weatherAUS.csv')
weather = weather.dropna()
weather['month'] = pd.DatetimeIndex(weather['Date']).month
weather = weather.drop(columns=['Date'])
X = weather.iloc[:,:-3]
X['month'] = weather['month']
Y1 = weather['RainToday']
Y2 = weather['RainTomorrow']
X = pd.get_dummies(X,columns = ['Location','WindGustDir','WindDir9am','WindDir3pm'])
fs = SelectKBest(score_func=f_classif, k=30)
fs.fit(X,Y1)
X_train_fs=fs.transform(X)

for i in range(len(fs.scores_)):
  print('Feature %d: %f' % (i, fs.scores_[i]))
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()
features = X.columns
top_features = []
for i in range(len(fs.scores_)):
  if fs.scores_[i] > 1000 :
    top_features.append(i)
print(len(top_features))
imp_features=[]
for i in top_features:
    imp_features.append(features[i])
print(imp_features)
X = X.iloc[:,top_features]
Y1 = makeLabel(Y1)
Y2 = makeLabel(Y2)
regressor = DecisionTreeRegressor(criterion='mse',random_state=100,max_depth=5,min_samples_leaf=30)
regressor.fit(X,Y1)
pickle.dump(regressor,open('model.pkl','wb'))
regressor2 = DecisionTreeRegressor(criterion='mse',random_state=100,max_depth=5,min_samples_leaf=30)
regressor2.fit(X,Y2)
pickle.dump(regressor,open('model2.pkl','wb'))
data = export_graphviz(regressor, out_file = 'regression_tree.dot')
data = export_graphviz(regressor, out_file = 'regression_tree2.dot')

