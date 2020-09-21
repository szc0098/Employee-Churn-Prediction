HR ANALYTICS OR PEOPLE ANALYTICS(A DATA DRIVEN APPROACH IN MANAGING PEOPLE AT WORK)

To know what type of problem we are dealing with or trying to predict
we can seperate into 2 categories:  Target = Churn or Turnover
                                    Features = Otherthan churn
Different problems approached are:
    Hiring/Assessment
    Retention
    Performance evaluation
Introduction:
    This is a project to predict the Employee churn(Employee attrition/ employee turnover), so that the 
    organization can take necessary steps to reduce the higher cost in employing new candidates and putting efforts not 
    to loose efficient employees.

Objective:
    From the past data we need to consider the attributes relevant to work with and perform some data manipulation and data
    preprocessing in order to develop a robust model for the prediction of Employee churn. Which features are 
    affecting more for the employee attrition.
    
Structure of data:
    here we have Categorical and Mathematical variables.
    
RangeIndex: 14999 entries, 0 to 14998
Data columns (total 10 columns):
satisfaction            14999 non-null float64
evaluation              14999 non-null float64
number_of_projects      14999 non-null int64
average_montly_hours    14999 non-null int64
time_spend_company      14999 non-null int64
work_accident           14999 non-null int64
churn                   14999 non-null int64
promotion               14999 non-null int64
department              14999 non-null object
salary                  14999 non-null object
dtypes: float64(2), int64(6), object(2)

This shows the variables of our dataset, last two columns show they are categorical variables(either ordinal-ranked or Nominal-No 
intrinsic order), 
so we have to work with department and salary attributes to convert them into numerical variables.
    
Nominal values
['sales' 'accounting' 'hr' 'technical' 'support' 'management' 'IT'
 'product_mng' 'marketing' 'RandD']

Ordinal values
['low' 'medium' 'high']

About the Predictive Model:
    Since it is a classification problem will the employee churn(leave the company or stay) classified primarity as 0 and
    1, it falls into binary classification. Since target is there it is a supervised classification.
    
    In Classification we have different types of models:
        a) Logistic Regression
        b) Decision Tree
        c) Deep Learning
        d) Neural Networks
        e) Support Vector Machines
    We are moving ahead with Decision Tree Model: last leaf will be split to get the pure sample for the model.
    DT uses 2 splitting rules:
        1) Entropy:
                    -p*log(p) - (1-p)*log(1-p)
        
        2) Gini:
                    2*p*(1-p)
        
Decision Tree: 
    It can be used for the classification and regression problem.(by the CART model). Where the tree can
    be divided and subdivided in subsets of data by establishing a proper linkage between the features 
    and the nodes to predict the final output as a value based from the given set of discrete values.
        
Overfitting:
    This is the most concerning problem when we work with decision trees. 
    Initial Score:
        Test score= 97.97333333333333

        Trainscore= 100.0

    How do we know if our model is overfitting:
        after training the test & train datasets and run the model if 
        Train shows Accuracy = 100%
        and when tested the same on Test Dataset Accuracy = 98 (or) 97 i.e comparatively less it means model is overfit.
        
    Solution:
        Limit the max_depth
        Limit the size of samples in leaves
    Later scores: 
        cross_val_score(model_depth_5, feature_test, target_test)*100
        array([97.76179057, 97.04      , 97.75820657])

        cross_val_score(model_sample_100, feature_test, target_test)*100
        array([95.36370903, 94.16      , 95.67654123])
        
        
Evaluating the Model:
    Prediction Errors:
        COnfusion Matrix:
            
            |____0____Actual____1____|
            |           |            |
         0  |     TN    |     FN     |
Predicted   |___________|____________|
            |           |            |
          1 |     FP    |     TP     |
            |___________|____________|
            
      Metrics 1:      
        If the target is leavers:(focus on FN)
            Recall Score = TP/(TP+FN)
            
        If the target is leavers:(focus on FP)
            Specificity = TN/(TN+FP)
        
     Metrics 2:       
        If the target is leavers: (focus on FP)
            Precision Score = TP/(TP+FP)
            
Imbalance in the Model:
    In the model the stayers are having higher percentage than the leavers. So the problem of imbalance is because of
    the higher weightage put by the stayers is 80%(so it predicts the stayers correctly) and leavers weightage is 20%
    (shows the probability of predicting leavers is not always correct)
    to correct this while building the model just balance the class weight.
    
Hyperparameter Tuning:
    previously by using the train/test split ensures that the training data is not overfitting
    it consists in tuning the model to get the best prediction results on the test set. Therefore, it is recommended 
    to validate the model on different testing sets. K-fold cross-validation allows us to achieve this:

    it splits the dataset into a training set and a testing set
    it fits the model, makes predictions and calculates a score (you can specify if you want the accuracy, precision, 
    recall...)
    it repeats the process k times in total
    it outputs the average of the scores
Set the parameters, choose the necessary features, here importance is defined to greater than 5%.
these defined features are fixed to the test and train datasets.

DecisionTreeClassifier is made fit to train and test components. Then the model is trained on the train dataset and predicted on the test set.

The final score is calculated to see the %difference between the predicted test values and the actual values.