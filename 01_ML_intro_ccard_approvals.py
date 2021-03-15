"""

# This script uses UCI Credit Approval data (http://archive.ics.uci.edu/ml/datasets/credit+approval)

"""

### 1. Load data and perform fist EDA
import pandas as pd

cc_apps = pd.read_csv('crx.data',header=None)

print(cc_apps.head())

# assign column names using a list
cc_apps.columns = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity',
                   'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'ApprovalStatus']

print(cc_apps.describe())    # Notice that only four columns are shown, the numeric ones. pass include='all' to see descriptive stats of non-numeric columns too

for col in cc_apps.columns:             # Let's see all columns. (How many unique values does 'Gender' have? What's the maximum age?)
    print(col)
    print(cc_apps[col].describe())
    print('\n')

print(cc_apps.info())

### 2. Inspecting and handling missing values

# 2.i looking for missing values
print(cc_apps.isnull().sum())       # It looks like there are no missing values. Or are there? We already hinted they may coded using the string '?'
print(cc_apps.tail(20))
print(cc_apps.ZipCode.value_counts()[:20]) # Check the 20 most frequent Zip codes. Is 00000 a valid ZipCode? Watch out for seemingly correct entries.

# 2.ii Replace the '?'s with np.NaN
import numpy as np
cc_apps = cc_apps.replace('?',np.NaN) 
print(cc_apps.isna().sum()) # Count the number of NaNs in the dataset
print(cc_apps.tail(20))     # Check the tail of the df to confirm the '?' have been changed to NaNs  

# 2.iii Imputing numeric cols with col means

# Check the numeric cols (Debt,  YearsEmployed, CreditScore and Income)
cc_apps.describe() # Looks like they need no imputation

cc_apps.iloc[0] # Now check the first row. Age and YearsEmployed are good candidates for imputation using the mean. We'll cast them as float afterwards

for col in ['Age','YearsEmployed']:
    cc_apps[col] = cc_apps[col].fillna(cc_apps[col].astype(float).mean()).astype(float)

for col in ['CreditScore','Income']:                # We also convert CreditScore and Income to floats, to simplify later use of continuous variables
    cc_apps[col] = cc_apps[col].astype(float)

# 2.iv Imputing non-numeric cols with their most frequent value
    
cc_apps.dtypes # Check column types

for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':                                 # Change only 'object'-type columns
        cc_apps[col] = cc_apps[col].fillna(cc_apps[col].value_counts().index[0])  # Impute with the most frequent value

### 3. 2nd EDA
# Explore the continuous variables/features using Seaborn's scatterplot matrix
import seaborn as sns

cont_features = list(cc_apps.loc[:,cc_apps.dtypes == float].columns)
sns.pairplot(data=cc_apps,hue='ApprovalStatus')     # By default, pairplot() will skip the object data types

# Scale the continuous features
from sklearn.preprocessing import StandardScaler,MinMaxScaler

scaler = MinMaxScaler()

cc_apps_scaled = cc_apps[['ApprovalStatus']].copy()
for col in cont_features:
    cc_apps_scaled[col] = scaler.fit_transform(cc_apps[col].values.reshape(-1, 1))
   
sns.pairplot(data=cc_apps_scaled,hue='ApprovalStatus')     # Build a scatterplot matrix with the logged features
# The scaler does not change the shape of the distributions, because it changes the scale only. Let's explore a transformation that changes the distributions' shapes: the log

# log the continuous features
cc_apps_logged = cc_apps[['ApprovalStatus']].copy()
for col in cont_features:
    cc_apps_logged['ln_{}'.format(col)] = np.log(cc_apps[col]+1)
   
sns.pairplot(data=cc_apps_logged,hue='ApprovalStatus')
# This way is more clear that people with age, years of employment, credit score, income and (even) debt tend to be higher for those whose credits are approved.

### 4. Baseline model: a logistic classifier

### 4.i Preprocessing data (i): Using LabelEncoder to transform non-numeric columns

cc_apps_encoded = cc_apps.copy() # We make a copy of cc_apps, where we will apply LabelEncoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cc_apps_encoded.columns.values:
    if cc_apps_encoded[col].dtypes=='object':    
        cc_apps_encoded[col]=le.fit_transform(cc_apps_encoded[col])     # Apply LabelEncoder. 

#Note that we encoded non-ordinal data, so that the ordering of the encoded variables is meaningless (e.g, marriage=2 is not necessarily 'greater' than marriage=1).
# The logistic classifier will run nevertheless, but we will use it with caution (it's but our simplest, baseline model).

### 4.ii Preprocessing data (ii): Splitting the dataset into train and test sets

from sklearn.model_selection import train_test_split

cc_apps_encoded = cc_apps_encoded.drop(['ZipCode'], axis=1)         # Drop the ZipCode feature (What's the rationale for this? Is this feature selection?)
cc_apps_colnames = cc_apps_encoded.columns[:-1]                     # Obtain the feature names
cc_apps_encoded = cc_apps_encoded.values                                              # and convert the df to a NumPy array by extracting its values
X, y = cc_apps_encoded[:,:14] , cc_apps_encoded[:,14]               # Separate features and labels into distinct variables
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=100)

### 4.iii. Fitting the model on the train set, making predictions and evaluating them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

logreg = LogisticRegression()       # Instantiate a logistic regressor
logreg.fit(X_train,y_train)         # Fit logreg to the (rescaled) train set. Notice that the default optimization algorithm (lbfgs) did not converge

print("Accuracy of the logistic regression classifier:\nOn training data: {}\nOn test data: {}".format(logreg.score(X_train,y_train),
                                                                                                       logreg.score(X_test,y_test)))      # Evaluate the performance of the baseline model

y_pred = logreg.predict(X_test)     # Make predictions
confusion_matrix(y_test,y_pred)     # Calculate the confusion matrix. How do off-diagonal elements look like?

# The baseline model did not converge. Often, logistic models fail to converge when the features vary wildly in scale as in this case. We will attack this issue by rescaling the features with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))             
rescaledX_train = scaler.fit_transform(X_train)         
rescaledX_test = scaler.fit_transform(X_test)

logreg_rescaled = LogisticRegression()       # Instantiate a logistic regressor to fit the rescaled model
logreg_rescaled.fit(rescaledX_train,y_train)
print("Accuracy of the logistic regression classifier:\nOn training data: {}\nOn test data: {}".format(logreg_rescaled.score(rescaledX_train,y_train),
                                                                                                       logreg_rescaled.score(rescaledX_test,y_test))) 
# The rescaled model converges. Notice that rescaling lowers the accuracy in training data while improving it on test data. Yet, this could be related to the particular train/test split used
# (other splits may produce different trends)

### 4.iv. Hyperparameter tuning: perform grid search on 'C' (the inverse of regularization strength, where smaller values specify stronger regularization)
# Notice that by default LogisticRegression uses l2 penalty (i.e, Ridge regularization)

print(LogisticRegression())     # Check the parameters of the logistic regression we instantiate through LogisticRegression()

from sklearn.model_selection import GridSearchCV

C = np.arange(.02,.4,.02)    # We specify a rather large grid for C
param_grid = dict(C=C)
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

rescaledX = scaler.fit_transform(X_train)                   # Notice we use only the train set in this 5-fold cross validation, we leave the test set to evaluate the cross-val result
                                                            # We can refert to this design as a train-validation-holdout setup
grid_model_result = grid_model.fit(rescaledX,y_train)

best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_     # Obtain the best score and the best parameter

print("The best score was: %f, obtained using %s" % (best_score, best_params))      # The best model implies quite a strong degree of regularization
                                                                                    # (this suggests we could have performed better in the feature selection step, as greater regularization essentially drops features)

grid_model_result.score(rescaledX_test,y_test)                  # Calculate the accuracy on the test (now holdout set) 
y_pred_best_model = grid_model_result.predict(rescaledX_test)   # Obtain predicted values in the test set using the best params (C=.04)

confusion_matrix(y_test,y_pred_best_model)                      # Calculate the confusion matrix. Notice that 'better' model essentially performs better wrt the detection of false positives
                                                                # 19 entries previously misclassified as approved are now correctly predicted as 'not approved'

# A sensible next step would be to use one-hot encoding on the categorical variables
grid_model_result.cv_results_
pd.DataFrame(zip(list(cc_apps_colnames),grid_model_result.best_estimator_.coef_[0]))    # Check out the model coefficients. In this case they are not illustrating, because of the encoding we used.
                                                                                        # They would be useful if we had one-hot encoded the categorical variables, though. Yet, even without it,
                                                                                        # we already achieved a somewhat decent accuracy (~83%) with a simple categorical encoding and minmax scaler.
