from sklearn import datasets,linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

space_titan_train = pd.read_csv("train_dim.csv")
space_titan_test = pd.read_csv('submission1226.csv')
space_titan_data = pd.read_csv('test_dim.csv')

space_titan_train_cat = space_titan_train.select_dtypes(object)
space_titan_train_num = space_titan_train.select_dtypes(np.number)
space_titan_train_bool = space_titan_train.select_dtypes(bool)

space_titan_test_cat = space_titan_test.select_dtypes(object)
space_titan_test_num = space_titan_test.select_dtypes(np.number)

space_titan_data_cat = space_titan_data.select_dtypes(object)
space_titan_data_num = space_titan_data.select_dtypes(np.number)

space_titan_train_cat.drop(['PassengerId','Name'],axis=1,inplace = True)

space_titan_test_cat.drop(['PassengerId'],axis=1,inplace = True)

space_titan_data_cat.drop(['PassengerId','Name'],axis=1,inplace = True)

space_titan_train_cat.HomePlanet.fillna(space_titan_train_cat.HomePlanet.value_counts().idxmax(),inplace=True)
space_titan_train_cat.CryoSleep.fillna(space_titan_train_cat.CryoSleep.value_counts().idxmax(),inplace=True)
space_titan_train_cat.Cabin.fillna(space_titan_train_cat.Cabin.value_counts().idxmax(),inplace=True)
space_titan_train_cat.Destination.fillna(space_titan_train_cat.Destination.value_counts().idxmax(),inplace=True)
space_titan_train_cat.VIP.fillna(space_titan_train_cat.VIP.value_counts().idxmax(),inplace=True)

space_titan_data_cat.HomePlanet.fillna(space_titan_train_cat.HomePlanet.value_counts().idxmax(),inplace=True)
space_titan_data_cat.CryoSleep.fillna(space_titan_train_cat.CryoSleep.value_counts().idxmax(),inplace=True)
space_titan_data_cat.Cabin.fillna(space_titan_train_cat.Cabin.value_counts().idxmax(),inplace=True)
space_titan_data_cat.Destination.fillna(space_titan_train_cat.Destination.value_counts().idxmax(),inplace=True)
space_titan_data_cat.VIP.fillna(space_titan_train_cat.VIP.value_counts().idxmax(),inplace=True)

space_titan_train_num.Age.fillna(space_titan_train_num.Age.mean(),inplace=True)
space_titan_train_num.RoomService.fillna(space_titan_train_num.RoomService.mean(),inplace=True)
space_titan_train_num.FoodCourt.fillna(space_titan_train_num.FoodCourt.mean(),inplace=True)
space_titan_train_num.ShoppingMall.fillna(space_titan_train_num.ShoppingMall.mean(),inplace=True)
space_titan_train_num.Spa.fillna(space_titan_train_num.Spa.mean(),inplace=True)
space_titan_train_num.VRDeck.fillna(space_titan_train_num.VRDeck.mean(),inplace=True)

space_titan_data_num.Age.fillna(space_titan_train_num.Age.mean(),inplace=True)
space_titan_data_num.RoomService.fillna(space_titan_train_num.RoomService.mean(),inplace=True)
space_titan_data_num.FoodCourt.fillna(space_titan_train_num.FoodCourt.mean(),inplace=True)
space_titan_data_num.ShoppingMall.fillna(space_titan_train_num.ShoppingMall.mean(),inplace=True)
space_titan_data_num.Spa.fillna(space_titan_train_num.Spa.mean(),inplace=True)
space_titan_data_num.VRDeck.fillna(space_titan_train_num.VRDeck.mean(),inplace=True)

le = LabelEncoder()

space_titan_train_cat = space_titan_train_cat.apply(le.fit_transform)
space_titan_train_bool = space_titan_train_bool.apply(le.fit_transform)

space_titan_data_cat = space_titan_data_cat.apply(le.fit_transform)

space_titan_train_full = pd.concat([space_titan_train_cat, space_titan_train_num,space_titan_train_bool],axis=1)


space_titan_test_full = space_titan_test_num

space_titan_data_full = pd.concat([space_titan_data_cat,space_titan_data_num],axis=1)

x = space_titan_train_full.drop(['Transported'],axis=1)
y = space_titan_train_full['Transported']

x_train = np.array(x[0:int(len(x))])
y_train = np.array(y[0:int(len(y))])

x_test = np.array(space_titan_data_full[0:int(len(space_titan_data_full))])
y_ans = np.array(space_titan_test_full[0:int(len(space_titan_test_full))])

RF = RandomForestClassifier()
RF_fit = RF.fit(x_train,y_train)

RF_pred = RF_fit.predict(x_test)
print(RF_pred)
print(accuracy_score(RF_pred,y_ans)*100)

RF_pred_proba = RF_fit.predict_proba(x_test)
print(RF_pred_proba)
