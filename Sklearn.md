## Sklearn
#### EDA
- df.shape
- df.info: get to know the variable type and row num
- df.describe: get to know the quantile value of each variable

#### Confusion Matrix
- why: to better evaluate the model performance because the tatget value is unbalanced and we need recall right to justicly evaluate the result.
- from sklearn.metrics import classification_report: directly get the report of precision recall f1-score.
- from sklearn.metrics import confusion_matrix: 

#### ROC curve
- Everything about threshold - select the best shreshold
- X:false positive rate
- Y: True positive rate
- Everypoint in the ROC is the peformance of each shreshold: upper left area is good.
- AUC is the area under the ROC, used for compare the different model. Different classification model represent the different AUC, the upper the better

```
# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr(false postive rate), tpr(true positive rate), thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

#### AUC : the area under the ROC
- using roc_auc_score(y_test, y_pred_prob) to get the AUC
- roc_auc (AUC score) is a scoring method that can be measure and compare the model

```
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg,X ,y, cv = 5,scoring = "roc_auc")

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

```

#### Hold-out set
- evaluate the model performance in a unseen data
- perform gridsearch cv on training set
- choose best hyper param and evaluate the model in hold-out set.
- the test set here will function as the hold-out set


```
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set (hold-out set) and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
```

#### Dummy variable
- convert categorical variable into numeric and then can be processed in SKlearn
- 0-1 variable 0:not in category, 1: in category
- pd.get\_dummies(df, drop_first=True) drop_first=True to drop the unneeded dummy variable
- if the original column have 5 category, then using get\_dummies can convert it into 4 different column with 0,1 value

```
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)
```

#### Missing data processing
- detect the meaningless 0 or ? value into some cols and convert them into np.nan
- then process the missing data in a approporiate way

```
df[df == "?"] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

```

#### Use imputer and pipeline to process the missing value
```
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values="NaN", strategy='most_frequent', axis=0)
#convert all the NaN value in to most_frequent(mode) value

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),   #first step: impute the data
        ('SVM', clf)]  
# second step: fit the model afterthe imputation


```
#### Create pipeline to fit the model

```

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
```

####Data scaling (normalize) in pipeline
- we want to features in a similar scale

- using noremal way:

```
# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

```
- scale the data into pipeline

```
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))
```

####Tuning pipeline
- build the steps including: imputer, scaler, estimator
- build the pipeline and put the steps into the pipeline
- establish the param dictionary
- GridSearchedCV using pipeline and param dictionary
- fit the GSCV

```
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]
         
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

```