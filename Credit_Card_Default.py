import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('UCI_Credit_Card.csv')
data.info()             # Checking the Null Entries

#Visualization

corr = data.corr()
plt.figure(figsize= (28,25))
sns.heatmap(corr, annot = True, vmin = -1.0, cmap = 'mako' )
plt.title(" Correlation HeatMap ")
plt.show()

# Preprocessing

{'EDUCATION': 'EDU'}.items()

def onehot_encode(df, column_dict) :
    df = df.copy()
    for column, prefix in column_dict.items() :         # will return tuples
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis = 1)
        df = df.drop(column, axis = 1)
    return df

def preprocess_input(df) :
    df = df.copy()

    df = onehot_encode(
        df,
        {
            "EDUCATION"  : "EDU",
            "MARRIAGE"   : "MAR"
        }
    ) 

    df = df.drop('ID', axis = 1)
    # Splitting into X and y

    X = df.drop('default.payment.next.month', axis = 1).copy()
    y = df['default.payment.next.month'].copy()

    # Scale X with a standard scaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X,y

X, y = preprocess_input(data)

# Training

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

models = {
    LogisticRegression(): "   Logistic Regression",
    SVC():                "Support Vector Machine",
    MLPClassifier():      "        Neural Network"
  
}

for model in models.keys():
    model.fit(X_train, y_train)

for model, name in models.items():
    print(name + ": {:.2f}%".format(model.score(X_test, y_test) * 100))