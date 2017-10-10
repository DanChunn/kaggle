import pandas as pd
import numpy as np
from sklearn import tree 

# Load the train and test datasets to create two DataFrames
train_path = "./data/train.csv"
train = pd.read_csv(train_path)

test_path = "./data/test.csv"
test = pd.read_csv(test_path)



#Adding "Child" column to our dataset
train["Child"] = float('NaN')
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0

# Fill in missing data
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
train["Child"] = train["Child"].fillna(0)

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# Create the target and features numpy arrays
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare", "Child"]].values
my_tree_one = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_one = my_tree_one.fit(features_one, target)



#Repeat for test
test["Child"] = float('NaN')
test["Child"][train["Age"] < 18] = 1
test["Child"][train["Age"] >= 18] = 0

# Convert the male and female groups to integer form
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# Fill in missing data
test["Age"] = test["Age"].fillna(train["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")
test["Child"] = test["Child"].fillna(0)
test["Fare"] = test["Fare"].fillna(train["Fare"].median())



# Predict
test_features = test[["Pclass", "Sex","Age", "Fare", "Child"]].values
my_prediction = my_tree_one.predict(test_features)

# Write to CSV
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("titanic_kaggle.csv", index_label = ["PassengerId"])
#print(my_solution)