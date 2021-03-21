# Modelling Library
Library containing the source code for easy model training and evaluation. 
The user inputs a configuration (json) file, which reads into a config class, saving all necessary modelling parameters.
A preprocessing pipeline features 
* NaN imputation
* Scaling of numerical variables
* Encoding of categorical variables
* SMOTE oversampling method.

Then, the modelling step is being added onto the preprocessing pipeline with XGBoost classifier or regressor (depending on the target listed in the configuration file). In the case of a multi-label classification problem, the sklearn MultiOutputClassifier is wrapped around the XGBoost estimator.
