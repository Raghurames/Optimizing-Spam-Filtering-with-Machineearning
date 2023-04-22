
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv', encoding = "ISO-8859-1")

# Clean data 

df.rename(columns={
    'v1': 'label',
    'v2': 'message',
}, inplace=True)

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Encode labels

le = LabelEncoder()
le.fit(df['label'])
df['label'] = le.transform(df['label'])

df.to_csv('models/saved/dataframe.csv', encoding='ISO-8859-1')

joblib.dump(le, 'models/saved/encoder.joblib')