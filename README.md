# bank_churn
Churn modelling project for bank customer dataset.

# Where is the code:
- You can see how the data was cleaned and analysed inside the notebook file  `churn_analysis.ipynb`:
  - A trained version of the best model was exported as `model.pkl`
- A .py file called `churn_predict.py` offers 2 useful functions:
  - `classify_pretrained()` loads the pre-trained model above, and predicts churn status.
  - `classify` loads the `BankChurners.csv` file, trains a classifier model on it, and predicts churn status. **(NOT FULLY IMPLEMENTED YET)**

# How to use
To classify a single customer record you need to create a clean conda environment using the `environments.yaml` file:
```bash
conda env create --name bank_churn_model --file environment.yml
conda activate bank_churn_model
```

Then you can open a python REPL with
```bash
python
>>> import pandas as pd
>>> customer = pd.DataFrame({'customer_age':45,
                             'gender': 'M',
                             'dependent_count':3,
                             'education_level':'High School',
                             'marital_status':,
                             'tenure_per_age':,
                             'utilisation_per_age':,
                             'credit_lim_per_age':,
                             'total_trans_amt_per_credit_lim':, 'total_trans_ct_per_credit_lim':,
                            })
>>> from app.clf_funcs import classify()
```


# ADDITIONAL QUESTIONS:
## Which machine learning models did you consider and why?
I considered the following models:
- **Logistic Regression**
- **SGD classifier (using 'log' loss)**
- **Decision Tree**
- **Random Forest**
- **XGBoost**

## What are the pros and cons of using each of these models?
- **Logistic Regression**.  This is a simple, explainable model.  Simply learns a series of +ve and -ve feature weights, which are intuitive to "explain" the relationships that the model has learned from the dataset.  Could be useful to identify critical features affecting churn rates.  This model also outputs a class probability prediction, which could be useful if we need to tweak decision thresholds if, for example we wanted to prioritise recall over precision.  Weaknesses are that it is a linear model, and can't learn complex interactions unless we introduce higher order features like polynomial terms etc.

- **SGD classifier (using 'log' loss)**.  This is basically a logistic regression but using stochastic gradient descent as the learning algorithm.  It computes the loss gradients over a subset of the data rather than over the whole dataset.  This makes it very computationally efficient, complexity is linear with the number of samples.  This model type could be useful for vastly larger datasets to allow out-of-core learning.  Disadvantages include that it is sensitive to feature scaling.

- **Decision Tree**.  Useful for baseline comparison. Not likely to learn a very accurate model using a single decision tree and it is prone to over-fitting, but it is highly explainable, doesn't require feature scaling and serves as foundation for the next few model types considered.  Weaknesses include that decision boundaries are always made of orthogonal lines, and that decision trees are easily biased by class imbalance.

- **Random Forest**.  An ensemble of many trees which reduces variance/overfitting and produces well generalised results.  Each individual tree in the "forest" can be trained in parallel, and the result is averaged over these.  Disadvantages are that random forests can be slower to train than other methods (was slowest in this project), but this will be sensitive to the number of trees etc as a hyperparameter.

- **XGBoost**.  A boosting method that trains a series of successive tree models, each one learning to minimise the residuals of the previous model.  This model is efficient to compute, considering the complexity of the relationships it learns.  I found it to produce high accuracy in it's base untuned form, and generalised well on testing data (and during cross validation).  Disadvantages include that as a tree-based method, it is not so effective for extrapolation.  Fortunately, in this problem of customer churn, extrapolation is not likely to be a strong focus of this model.

## What criteria did you use to decide which model was the best?
I considered metrics:
- Accuracy
- precision/recall
- F1 score
- AUC
- Mean and std_dev of cross validation scores (accuracy)
In this project I mainly looked for good f1 scores, since the plain *accuracy* metric is heavily skewed by the class imbalance in the raw data.

## Would you change your solution in any way if the input dataset was much larger (e.g. 10GB)?
If I was prioritising computation efficiency on a much larger dataset I would consider tuning the stochastic gradient descent model further as it should be possible to make it just as good as regular Logistic Regrssion, and with it's linear time complexity, was the most efficient to train.

However XGBoost is particularly optimised for fast, parallelisable training so I would try it's inbuilt options to handle out-of-core learning.  It could be trained on a cluster of multiple computers to aid training time, though 10Gb is probably near the threshold where distributed computing just starts to become worthwhile. 

If the dataset was so large, I would also consider trying a deep-learning method.  Neural networks can learn very complex relationships from highly dimensional data, but work best when the dataset is much larger (than say the provided 10,000 records).

Also, if the dataset was 10Gb I would prefer to store and query it from a SQL  database etc.  CSV files are useful and highly portable, but are not optimised for large datasets like SQL databases are, which would make data cleaning and exploraion much easier at that scale.

## How would you approach this challenge if the input dataset could not fit in your device memory?
As previous, I would look to employ a parallelisable/distributable model that allows out-of-core learning.  XGBoost could work for this, while also being the highest performing model tested.  (with more time I would have tried tuning each model, but so far have only considered baseline performance with default hyperparameters).

Again, I would look to store and query the data from something like a SQL database.  

Pandas dataframes require to be loaded completely into memory at once, so if the dataset simply did not fit into memory, I would need to approach the way data is loaded differently.
1. "chunked" loading of dataset, using generator functions etc to read data in line by line etc
2. Try an alternative library such as `PySpark` or `Dask`.  These enable "lazy" computation, where data is loaded and processed only as much as needed.  This makes it possible to process datasets bigger than the available RAM.  PySpark and Dask are also suitable for distributed computing across a cluster etc.

## What other considerations would need to be taken into account if this was real customer data?
If this was real customer data, I would want to consider the source of data and whether extra measures should be taken to anonymise, protect privacy in the dataset.

I would also want to consider factors like other demographic, geographic influences on the biases in this dataset.  This data looks like it is in dollars so probably covers US customers.  The classifier trained on this data might not generalise to other regions etc.

From an AI ethics point of view, I would want to ensure that the model does not inadvertently start penalising people unfairly, based on particular demographics or operator induced biases.  In this project however it seems that the purpose is only for predicting churn, so the negativeconsequences of incorrect predictions are probably relatively minor.