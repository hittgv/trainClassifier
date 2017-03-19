from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib

# read it as a rowwise dataset and lable it as fake
data_fake = pd.read_json('fake.json').transpose()
data_fake['fake_flag'] = True

data_real = pd.read_json('real.json').transpose()
data_real['fake_flag'] = False

all_data = data_fake.append(data_real)

print all_data

X = all_data.drop('fake_flag', axis=1).drop('url', axis=1)
y = all_data['fake_flag']

model = LogisticRegression()


model.fit(X, y)

print(model.get_params())

expected = pd.DataFrame(y)

predicted = pd.DataFrame(model.predict(X))

probs = model.predict_proba(X)

print(probs)

joblib.dump(model, 'fakeNews.pkl')

# print(pd.concat([expected, predicted]))

# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))

# print(model.score(X, y))
