# mlnotebook
# ML Learning Notebook
### Ensemble Method Building ML Model
- Voting model 
    - hard voting (mode) - only for classification
    - soft voting (prob) - regression or classification
        - Averaging model

- Bagging (combination)
    - using base model (combine together)
- Bagging (self-revise)
    - AdaBoost (sequential)
    - Gradient Boosting
        - XGBoost (parallel)
        - LightBGM (parallel)
        - CatBoost
- Stacking (relay races)
    - Hand made using Sklearn
        - prepare dataset
        - build first layer of estimators
        - append predictions to the dataset
        - build second layer meta estimator
        - use the stacked model for prediction
    - MLxtend
        - A package can be directly used for building stacking model
