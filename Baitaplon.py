import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

TRAIN_PATH = "D:\BTL-Nhom8\Train.xlsx"
TEST_PATH = "D:\BTL-Nhom8\Test.xlsx"
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Load the data
df = pd.read_excel(TRAIN_PATH)

# Define a function to copy the dataframe
def copy(df):
    return df.copy()

# Define a function to drop a column
def drop(df, col):
    new_df = df.drop(col, axis=1)
    return new_df

# Define a function to drop rows based on a condition
def drop_row(df, col, find):
    new_df = df[df[col] != find]
    return new_df

def get_dummies(df, col):
    return pd.get_dummies(df, columns=[col])
    
def replace(df, col):
    replace_list = [",", " minutes"]
    new_df = df.copy()
    for rep in replace_list:
        new_df[col] = new_df[col].str.replace(rep, "")
    return new_df

def to_numeric(df, col):
    new_df = df.copy()
    new_df[col] = pd.to_numeric(new_df[col])
    return new_df

def train_pipeline(df):
    return (df.pipe(replace, "Delivery_Time")
              .pipe(to_numeric, "Delivery_Time"))

# Define a function for the data preprocessing pipeline
def pipeline(df):
    return (df.pipe(copy)
              .pipe(get_dummies, "Location")
              .pipe(to_numeric, "Distance")
              .pipe(drop_row, "Cost", "for")  
              .pipe(to_numeric, "Cost"))

train = pipeline(df)
train = train_pipeline(train)
train

train_data = drop(train, "Delivery_Time")
train_data

train_data_numpy = train_data.to_numpy()
train_data_numpy

train_label = train["Delivery_Time"]
train_label

# Create and train the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(train_data_numpy, train_label)

# Evaluate the Linear Regression model using cross-validation
linear_reg_scores = cross_val_score(linear_reg, train_data_numpy, train_label, scoring="neg_mean_squared_error", cv=10)
linear_reg_rmse_scores = np.sqrt(-linear_reg_scores)

# Define a function to display the evaluation scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Display the evaluation scores for Linear Regression
display_scores(linear_reg_rmse_scores)

# Load the test data
test_df = pd.read_excel(TEST_PATH)

# Preprocess the test data
dataset = pd.concat([df, test_df], axis=0)
dataset = pipeline(dataset)
test = dataset[len(train):]
test = test.drop("Delivery_Time", axis=1)

# Make predictions using Linear Regression
prediction = linear_reg.predict(test)

# Create a submission DataFrame
submission = pd.DataFrame({"Delivery_Time": prediction})
submission.to_csv('result.csv', index=False)
