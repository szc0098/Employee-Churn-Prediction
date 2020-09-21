'''
    MODEL FOR PREDICTING THE EMPLOYEE CHURN TO TAKE NECESSARY ACTIONS IN ADVANCE TO AVOID THE COST OF LOSING THE EMPLOYEE.  
'''


import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sb
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import tree

f_read = pd.read_csv("churn_year.csv")

#Looking at the types of attributes of the dataset
print(f_read.info())
f_read.head()

#Looking for the types of departments and salary
print(f_read.department.unique())

print(f_read.salary.unique())

#Transforming the categorical variables into numerical variables
f_read.salary = f_read.salary.astype('category')

f_read.salary = f_read.salary.cat.reorder_categories(['low', 'medium', 'high'])
          

#Assigning numerical format to order 
f_read.salary = f_read.salary.cat.codes
print(f_read.salary)

#Creating dummies and storing in another dataframe for Nominal category variable 
nom_dept = pd.get_dummies(f_read.department)
print(nom_dept)

#here we observe the dummy trap in the column where it shows 1 (here its 'sales' column in department_all dataframe.)
'''

department_all.head()
Out[12]: 
   IT  RandD  accounting  hr  management  marketing  product_mng  sales  \
0   0      0           0   0           0          0            0      1   
1   0      0           0   0           0          0            0      1   
2   0      0           0   0           0          0            0      1   
3   0      0           0   0           0          0            0      1   
4   0      0           0   0           0          0            0      1   

   support  technical  
0        0          0  

'''

nom_dept = nom_dept.drop("sales", axis = 1)
nom_dept.head()

'''
department_all.head()
Out[14]: 
   IT  RandD  accounting  hr  management  marketing  product_mng  support  \
0   0      0           0   0           0          0            0        0   
1   0      0           0   0           0          0            0        0   
2   0      0           0   0           0          0            0        0   
3   0      0           0   0           0          0            0        0   
4   0      0           0   0           0          0            0        0   

   technical  
0          0  

'''

#Droping the department column from the data table and inserting department_all data into data table

f_read = f_read.drop("department", axis = 1)
print(f_read.info())


#Joining 2 tables
f_read = f_read.join(nom_dept)
f_read.tail()

'''
Now lets deep dive into data we need to predict the churn of employees. following steps calculate the percentage of
churn for the given datapoints.

'''

cal_churn_val = (f_read.churn.value_counts())/len(f_read)*100

print(cal_churn_val)

#Using the visualization libraries in pandas 
correlation_matrix = f_read.corr()
sb.heatmap(correlation_matrix)
plot.show()


#Developing a model: Preparing the variables and splitting data into test and train datasets

print(f_read.head())
y = f_read.churn
X = f_read.drop("churn", axis = 1)

from sklearn.model_selection import train_test_split
y_train, y_test, X_train, X_test = train_test_split(y, X, train_size = 0.80)

#Implementation of Decision Tree Model 
from sklearn.tree import DecisionTreeClassifier

#Random state is used so that if we run it next time also we will get the same results
classifier = DecisionTreeClassifier(random_state = 50) 

#Train dataset to learn the model
classifier.fit(X_train, y_train)

#Testing the accuracy with the test data.
classifier.score(X_train, y_train)*100
classifier.score(X_test, y_test)*100

# Import the graphical visualization export function
from sklearn.tree import export_graphviz

# Exporting the tree to a dot file
export_graphviz(classifier,"tree.dot")

#Tuning employee churn classifier this is the solution to overfitting of the model
#Pruning the Model by selecting the best number for max_depth:
model_depth_5 = DecisionTreeClassifier(max_depth = 5, random_state = 42)

#To avoid overfitting here we are defining the min samples per leaf to 100, this is other way
model_sample_100 = DecisionTreeClassifier(min_samples_leaf = 100, random_state = 42)

#Theresults show that the variance between the test and train is relative less, i.e., overfitting is avoided.
cross_val_score(model_depth_5, X_test, y_test)*100

cross_val_score(model_sample_100, X_test, y_test)*100

# Setting all parameters to develop a model
mod_1 = DecisionTreeClassifier(max_depth=7, class_weight="balanced", random_state=42)
# fit the model on the train dataset 
mod_1.fit(X_train,y_train)
# Predicting on the test set componentprediction_b = mod_1.predict(X_test)
# Printing the recall score for the predicted model
recall_score(y_test,prediction_b)
# Printing the ROC/AUC score for the predicted model
roc_auc_score(y_test, prediction_b)

#Reducing the problem of Hyperparameter tuning is by introducing the k-fold cross vallidation.
from sklearn.model_selection import cross_val_score
cross_val_score(classifier,X,y,cv=15)

# import the GridSearchCV function
from sklearn.model_selection import GridSearchCV

# Getting the values for maximum depth. As we change the range accuracy of the model varies
#It varies from model to model based on data and prediction we are choosing
depth = [i for i in range(5,30,3)]

# Gives the values for minimum sample size, as we change the range accuracy of the model varies
samples = [i for i in range(5,500,50)]

# Creating the dictionary with parameters to be checked
parameters = dict(max_depth=depth, min_samples_leaf=samples)

# Setting up parameters: done
parameters = dict(max_depth=depth, min_samples_leaf=samples)
  
# Initializing the param object using the GridSearchCV function, initial model and parameters above
param = GridSearchCV(classifier, parameters)
print(param)

# Fitting the param_search to the training dataset
print(param.fit(X_train, y_train))


# Calculating feature importances
X_importances = classifier.feature_importances_

# Creating a list of features
X_list = list(X)

# Saving the results inside a DataFrame using feature_list as an index
DT_importance = pd.DataFrame(index=X_list, data=X_importances, columns=["importance"])

# Sorting values to learn most important features
DT_importance.sort_values(by="importance", ascending=False)

# Selecting only features with importance higher than 5%
final_X = DT_importance[DT_importance.importance>0.05]
print(final_X)
# final list of features is created
final_list = final_X.index

# Transforming both features_train and features_test components to include only selected features
features_train_selected = X_train[final_list]
features_test_selected = X_test[final_list]


# As per the best parameters given above model has been initialized
model_best = DecisionTreeClassifier(max_depth=10, min_samples_leaf=150, class_weight="balanced", random_state=50)

# Fitting the model using only selected features from training set
model_best.fit(features_train_selected, y_train)

# Making prediction based on selected list of features from test set
prediction_best = model_best.predict(features_test_selected)

# Printing the general accuracy of the model_best
print(model_best.score(features_test_selected, y_test) * 100)

'''
#Final visualization
visual_graph = tree.export_graphviz(model_best, out_file='tree.dot', filled=True, rounded=True,
                                feature_names=final_list,  
                                class_names="churn")
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import pydot
dot_data = export_graphviz(model_best, out_file=None,
                filled=True, rounded=True,
                special_characters=True)
graph = pydot.graph_from_dot_data(dot_data)
dot_data(graph.create_png())

'''




























