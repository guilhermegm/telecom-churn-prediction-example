from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# fix random seed for reproducibility
np.random.seed(7)

def normalize_dataset():
  dataset = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')

  dataset['TotalCharges'] = dataset['TotalCharges'].replace(' ',np.nan)
  dataset = dataset[dataset['TotalCharges'].notnull()]
  dataset = dataset.reset_index()[dataset.columns]
  dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)

  replace_cols = [
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'PaperlessBilling',
    'Churn',
  ]
  for i in replace_cols:
    dataset[i]  = dataset[i].replace({ 'Yes': 1, 'No': 0, 'No internet service': 2, 'No phone service': 2 })

  dataset['gender'] = dataset['gender'].replace({ 'Female': 0, 'Male': 1 })
  dataset['PaymentMethod'] = dataset['PaymentMethod'].replace({
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3,
  })
  dataset['Contract'] = dataset['Contract'].replace({
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
  })
  dataset['InternetService'] = dataset['InternetService'].replace({
    'No': 0,
    'DSL': 1,
    'Fiber optic': 2,
  })

  dataset = dataset.drop(columns = 'customerID', axis = 1)

  Y = dataset['Churn'].values
  dataset = dataset.drop(columns = 'Churn', axis = 1)
  X = dataset.values

  return X, Y


X, Y = normalize_dataset()

model = Sequential()
model.add(Dense(12, input_dim=19, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10)

scores = model.evaluate(X, Y)
print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

preds = model.predict(X)
print([(r[0] and 'Yes' or 'No') for r in model.predict_classes(X)[:100]])
