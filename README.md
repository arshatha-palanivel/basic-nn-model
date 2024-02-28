# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![DL 01](https://github.com/arshatha-palanivel/basic-nn-model/assets/118682484/60a424cf-58a6-41a5-a6c4-1339f1828829)



## DESIGN STEPS

### STEP 1:

Loading the dataset.

### STEP 2:

Split the dataset into training and testing.

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot.

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: ARSHATHA P
### Register Number: 212222230012
### Dependencies:
```py
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```
### Data From Sheets:
```py
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DATA').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```
### Data Visualization:
```py
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df
x=df[['INPUT']].values
y=df[['OUTPUT']].values
```
### Data split and Preprocessing:
```py
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train1=scaler.transform(x_train)
```
### Regressive Model:
```py
ai_brain = Sequential([
    Dense(6,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(x_train1,y_train,epochs = 1000)
```
### Loss Calculation:
```py
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```
### Evaluate the model:
```py
x_test1 = scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)
```
### Prediction:
```py
x_n1 = [[5]]
x_n1_1 = scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```
### Dataset Information
![DL 02](https://github.com/arshatha-palanivel/basic-nn-model/assets/118682484/c6fa8665-71ba-486d-9fd6-7813ea6fc7db)


## OUTPUT

### Training Loss Vs Iteration Plot
![DL 04](https://github.com/arshatha-palanivel/basic-nn-model/assets/118682484/ad543a4a-4c7e-403c-93b9-33523f29a27c)

### Training
![DL 03](https://github.com/arshatha-palanivel/basic-nn-model/assets/118682484/5d853f17-cf2d-48f4-826c-863b14fb6a99)


### Test Data Root Mean Squared Error
![DL 05](https://github.com/arshatha-palanivel/basic-nn-model/assets/118682484/82933ce2-22bd-4ef9-ae0f-14c2df676171)


### New Sample Data Prediction
![DL 06](https://github.com/arshatha-palanivel/basic-nn-model/assets/118682484/96ff24c2-5373-4d77-bd1e-15efcf584cf4)



## RESULT

A neural network regression model for the given dataset has been developed Sucessfully.
