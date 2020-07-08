### ML Workflow
####Feature engineering - Label encoding and One-hot
- First define the LabelEncoder
- Second transfer it into every column using fit_transform(col)
- check the after-converted data type to see if the non0numeric value has been converted into numeric value

```
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])
    
#Inspect the data types of the columns of the data frame
print(credit.dtypes)
```
- label encoder just convert the diferent categories into different numberic number
- One-hot encoding convert the different categories into different columns of dummy variable
- use .fit_transform to apply the encoder to specific column

```

# Create numeric encoding for credit_history
credit_history_num = LabelEncoder().fit_transform(credit['credit_history'])

# Create a new feature matrix including the numeric encoding
X_num = pd.concat([X, pd.Series(credit_history_num)], 1)

# Create new feature matrix with dummies for credit_history
X_hot = pd.concat(
  [X, pd.get_dummies(credit['credit_history'])], 1)

# Compare the number of features of the resulting DataFrames
print(X_hot.shape[1] > X_num.shape[1])
```

#### Feature selection using SelectKBest()
- we might need to transform some feature into a new features
- then, we want to select the best feature that have close relationship with y
- Score function:
	- For regression: f_regression, mutual\_info_regression
	- For classification: chi2, f\_classif, mutual\_info\_classif

```
# Function computing absolute difference from column mean
def abs_diff(x):
    return np.abs(x-np.mean(x))

# Apply it to the credit amount and store to new column
credit['diff'] = abs_diff(credit['credit_amount'])

# Create a feature selector with chi2 that picks one feature
sk = SelectKBest(chi2, k=1) 
#chi2 is the determine rule, k is the number feature we finally want to select.

# Use the selector to pick between credit_amount and diff
sk.fit(credit[['credit_amount', 'diff']], credit['class'])
# the two arguments in the sk.fit is X(the candidate vairbales) and y

# Inspect the results
sk.get_support()
```
```
# Find the best value for max_depth among values 2, 5 and 10
grid_search = GridSearchCV(
  rfc(random_state=1), param_grid={'max_depth':[2,5,10]})
best_value = grid_search.fit(
  X_train, y_train).best_params_['max_depth']

# Using the best value from above, fit a random forest
clf = rfc(
  random_state=1, max_depth=best_value).fit(X_train, y_train)

# Apply SelectKBest with chi2 and pick top 100 features
vt = SelectKBest(chi2, k=100).fit(X_train, y_train)

# Create a new dataset only containing the selected features
X_train_reduced = vt.transform(X_train)
```

#### Data Fusion
- using groupby function with the function you created to reconstruct the 

```
#create a function first, to output the column you like

def featurize(df):
return {
'unique_ports': len(set(df['destination_port'])),
'average_packet': np.mean(df['packet_count']),
'average_duration': np.mean(df['duration'])
}


#Group by source computer, and apply the feature extractor

out = flows.groupby('source_computer').apply(featurize)

#The output is a group-by dictionary

# Convert the iterator to a dataframe by calling list on it
X = pd.DataFrame(list(out), index=out.index)
#source_computer is the index because it is followed by groupby

# Check which sources in X.index are bad to create labels
y = [x in bads for x in X.index]

# Report the average accuracy of Adaboost over 3-fold CV
print(np.mean(cross_val_score(AdaBoostClassifier(), X, y)))
```
#### Using lambda function and groupby to create the new df

```
# Create a feature counting unique protocols per source
protocols = flows.groupby('source_computer').apply(
  lambda df: len(set(df['protocol'])))
  
#using len(set(columns) to identify the number of unique value
#set() return the unique value of the list

# Convert this feature into a dataframe, naming the column
protocols_DF = pd.DataFrame(
  protocols, index=protocols.index, columns=['protocol'])

# Now concatenate this feature with the previous dataset, X
X_more = pd.concat([X, protocols_DF], axis=1)

# Refit the classifier and report its accuracy
print(np.mean(cross_val_score(
  AdaBoostClassifier(), X_more, y)))
```

####Deal with imperfect label (some label might be wrong)
- rebuild the label based on some shresholds of the vairbale 
- refit the model the re-evaluate the model after we modify the label

```
# Create a new dataset X_train_bad by subselecting bad hosts
X_train_bad = X_train[y_train]

# Calculate the average of unique_ports in bad examples
avg_bad_ports = np.mean(X_train_bad['unique_ports'])

# Label as positive sources that use more ports than that
pred_port = X_test['unique_ports'] > avg_bad_ports

# Print the accuracy of the heuristic
print(accuracy_score(y_test, pred_port))

```
- using two shresholds to recreate the label
- use the principle： True*True = 1 to define the new label

```
# Compute the mean of average_packet for bad sources
avg_bad_packet = np.mean(X_train[y_train]['average_packet'])

# Label as positive if average_packet is lower than that
pred_packet = X_test['average_packet'] < avg_bad_packet

# Find indices where pred_port and pred_packet both True
pred_port = X_test['unique_ports'] > avg_bad_ports
pred_both = pred_packet * pred_port #True*True=1, False*False=0, False*True = 0

# Ports only produced an accuracy of 0.919. Is this better?
print(accuracy_score(y_test, pred_both))
```

#### Weighted Learning (need to be further learned)

```
# Fit a Gaussian Naive Bayes classifier to the training data
clf = GaussianNB().fit(X_train, y_train_noisy)

# Report its accuracy on the test data
print(accuracy_score(y_test, clf.predict(X_test)))

# Assign half the weight to the first 100 noisy examples
weights = [0.5]*100 + [1.0]*(len(y_train_noisy)-100)

# Refit using weights and report accuracy. Has it improved?
clf_weights = GaussianNB().fit(X_train, y_train_noisy, sample_weight=weights)
print(accuracy_score(y_test, clf_weights.predict(X_test)))
```

#### Loss Function - Create new scoring method
- Loss function is the rule to evaluate the model

Split the tn, fp, fn and tp using confusion_matrix and ravel()
Then build the new loss function using tn fp fn and tp

```
# Get false positives/negatives from the confusion matrix
tn, fp,fn , tp = confusion_matrix(y_test, preds).ravel()

# Now compute the cost using the manager's advice
cost = fp*10 + fn*150    #That is a new loss function
```
#### Adjust shreshold is also a method to adjust the loss_function
- calculate the probability score of getting the target value
- create the range of shreshold candidates
- get the different predictive value based on the different shreshold value
- get the different score of the different predictive value with the different shreshold, and select the best one with the best shreshold

```
# Score the test data using the given classifier
scores = clf.predict_proba(X_test)

# Create a range of equally spaced threshold values
t_range = [ 0.0, 0.25, 0.5, 0.75, 1.0]

# Store the predicted labels for each value of the threshold
preds = [[s[1] > thr for s in scores] for thr in t_range]

# Compute the accuracy for each threshold
accuracies = [accuracy_score(y_test, p) for p in preds]

# Compute the F1 score for each threshold
f1_scores = [f1_score(y_test, p) for p in preds]

# Report the optimal threshold for accuracy, and for F1
print(t_range[argmax(accuracies)], t_range[argmax(f1_scores)])
```
### Pipeline with different steps and different params of each steps
- define a new scorer
- create a pipeline including the several steps you want to process the data.
	- imputer: Imputer(missing_values="NaN",strategy='most_frequent', axis=0)
	- scaler: StandardScaler()
	- feature selection：  SelectKBest(f_classif)
	- model fitting
- create a dictionary of the parameters in different step
- using __ to link the key name and step name
- GirdSearch the pipe with the pipeline parameters and use new-defined scorer in the scoring arguemnt.
- fit the pipeline using X_train and y_trian
- using .best_params to get the best parameters of the model

```
# Create a custom scorer
scorer = make_scorer(f1_score)

# Create pipeline with feature selector and classifier
pipe = Pipeline([
    ('feature_selection', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier(random_state=2))])

# Create a parameter grid
params = {
   'feature_selection__k':[10, 20],
    'clf__n_estimators':[2, 5]}

# Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid=params, scoring = scorer)

# Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)
```

#### Deploy the model
- you can create your own pipeline steps for yourself by defining a function
- put your new function into the pipeline
- define your pileline parameters
- Girdsearch the pipeline with the defined parameters
- fit the girdsearch to build the model
- write the model into the file 
- load the model from the file when you want to use the model

```
# Define a feature extractor to flag very large values (a new steps)

def more_than_average(X, multiplier=1.0):
  Z = X.copy()
  Z[:,1] = Z[:,1] > multiplier*np.mean(Z[:,1])
  return Z

# Convert your function so that it can be used in a pipeline
pipe = Pipeline([
  ('ft', FunctionTransformer(more_than_average)),
  ('clf', RandomForestClassifier(random_state=2))])

# Optimize the parameter multiplier using GridSearchCV
params = {'ft__multiplier':[1,2,3]}
grid_search = GridSearchCV(pipe, param_grid=params, cv = 3)
gs = grid_search.fit(X_train, y_train)

# Save it to a file, to be pushed to production
with open('model.pkl', 'wb') as file:   #write
    pickle.dump(gs, file=file)

# Now load the model from file in the production environment
with open('model.pkl',"rb") as file:   #read
    gs_from_file = pickle.load(file)

# Predict the labels of the test dataset
preds = gs_from_file.predict(X_test)
```
### Detecting overtting
- CV Training Score >>CV Test Score
	- overtting in model tting stage
	- reduce complexity of classier
	- get more training data
	- increase cv number
- CV Test Score >> Validation Score
	- overtting in model tuning stage
	- decrease cv number
	- decrease size of parameter grid

```
# Fit your pipeline using GridSearchCV with three folds
grid_search = GridSearchCV(
  pipe, params, cv=3, return_train_score=True)

# Fit the grid search
gs = grid_search.fit(X_train, y_train)

# Store the results of CV into a pandas dataframe
results = pd.DataFrame(gs.cv_results_)

# Print the difference between mean test and training scores
print(
  results['mean_test_score']-results['mean_train_score'])
```
### Dataset shift
- when the production data has slightly different difference with the model training dataset, then the pre-trained model will have a bad performance on the shifted data because the original model did not cpature the pattern of the new data.
- reason of the dataset shift:
	- temporary change
	- domain shift: eg: the range of people that data was collected from is changed. 
- What to do?
	- retrain the model using part of data using the Window size
	
```
# Loop over window sizes
for w_size in wrange:

    # Define sliding window
    sliding = arrh.loc[(t_now - w_size + 1):t_now]

    # Extract X and y from the sliding window
    X, y = sliding.drop('class', 1), sliding['class']
    
    # Fit the classifier and store the F1 score
    preds = GaussianNB().fit(X, y).predict(X_test)
    accuracies.append(f1_score(y_test, preds))

# Estimate the best performing window size
optimal_window = wrange[np.argmax(accuracies)]
```

```
# Create a pipeline 
pipe = Pipeline([
  ('ft', SelectKBest()), ('clf', RandomForestClassifier(random_state=2))])

# Create a parameter grid
grid = {'ft__k':[5, 10], 'clf__max_depth':[10, 20]}

# Execute grid search CV on a dataset containing under 50s
grid_search = GridSearchCV(pipe, param_grid=grid)
arrh = arrh.iloc[np.where(arrh['age'] < 50)]
grid_search.fit(arrh.drop('class', 1), arrh['class'])

# Push the fitted pipeline to production
with open('pipe.pkl', 'wb') as file:
    pickle.dump(grid_search, file)
    
```
#### Unsupervised ML
- negative outlier factor: larger value means more normal
- contamination: how many percent of the outlier you think in the dataset


```
# Import the LocalOutlierFactor module
from sklearn.neighbors import LocalOutlierFactor as lof

# Create the list [1.0, 1.0, ..., 1.0, 10.0] as explained
x = [1]*30
x.append(10)

# Cast to a data frame
X = pd.DataFrame(x)

# Set the contamination parameter to 0.2
preds = lof(contamination=0.2).fit_predict(X)

# Print the confusion matrix
print(confusion_matrix(ground_truth,preds ))
```

