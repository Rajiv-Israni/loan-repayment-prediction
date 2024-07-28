# loan-repayment-prediction

- In this project we predict if a customer requesting for loan will be able to pay it, or will default on it.
- The algorithm which we used is Random Forest Classifier and we applied over the given dataset to increase the performance as well as to achieve highest accuracy.
- A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
- The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).
- The main reason was to divide transaction columns to debit and credit into multiclass array format, while they are the weak learners and the role of the random forest(by voting) is to amend weak learners to strong learners.
- In this dataset to achieve the highest output we also applied hyper parameter optimization to select the best estimators. Hyper-parameters are parameters that are not directly learnt within estimators. In scikit-learn they are passed as arguments to
  the constructor of the estimator classes. Typical examples include C, kernel and gamma for Support Vector Classifier, alpha for Lasso, etc.
- It is possible and recommended to search the hyper-parameter space for the best cross validation score.
- With this we achieved 73% accuracy by applying best estimators too.
