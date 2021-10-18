# bank_churn
Churn modelling project for bank customer dataset.

# Where is the code:
- You can see how the data was cleaned and analysed inside the notebook file  `churn_analysis.ipynb`:
  - A trained version of the best model was exported as `model.pkl`
  - A link to the notebook is [link on nbviewer.org](https://nbviewer.org/github/adin786/bank_churn/blob/main/churn_analysis.ipynb)
- A .py file called `churn_predict.py` offers 2 useful functions:
  - `classify_pretrained()` loads the pre-trained model above, and predicts churn status.
  - `classify` loads the `BankChurners.csv` file, trains a classifier model on it, and predicts churn status. **NOT FULLY IMPLEMENTED YET)**

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
                            'tenure_per_age utilisation_per_age  credit_lim_per_age  total_trans_amt_per_credit_lim  total_trans_ct_per_credit_lim})
>>> from churn_predict import classify()
```


# ADDITIONAL QUESTIONS:
## Which machine learning models did you consider and why?
I considered:
- **Logistic Regression**.  Simple, explainable model, could be useful to investigate feature weights/importances
- **SGD classifier (using 'log' loss)**.  Basically a logistic regression but using stochastic gradient descent as the learning algorithm.  This makes it extremely computationally efficient and parallelisable.  This model could be useful for vastly larger datasets to allow out-of-core learning.
- **Decision Tree**.  Useful for baseline comparison. Not likely to learn a very accurate model using a single decision tree and it is prone to over-fitting, but it is highly explainable, and serves as foundation for the next few model types considered.
- **Random Forest**.  An ensemble of many trees which reduces overfitting.  Good accuracy, and parallelisable, but relatively slow to train.
- **XGBoost**.  A boosting method that trains a series of successive tree models, each one learning to minimise the residuals of the previous model.  This model is highly efficient to compute, considering the complexity of the relationships it learns.  I found it to produce high accuracy in it's base untuned form, and generalised well on testing data (assessed using cross validation).

## What are the pros and cons of using each of these models?
As described above.

## What criteria did you use to decide which model was the best?
I considered metrics:
- Accuracy on training and testing sets
- precision, recall, mainly looking at f1
- AUC
- Mean of cross validation scores (accuracy)
- Standard deviation of the above cross validation scores

## Would you change your solution in any way if the input dataset was much larger (e.g. 10GB)?
If I was prioritising computation efficiency on a much larger dataset I would consider tuning the stochastic gradient descent model further as it was most efficient in my training.

However XGBoost is specifically designed for fast, parallelisable computation so I would try it's inbuilt options to handle out-of-core learning.  It could be trainied on GPUs, or on a cluster if memory became an issue. 

If the dataset was so large, I would also consider trying a deep-learning method.  Neural networks can learn very complex relationships from highly dimensional data, but work best when the dataset is much larger (than say the provided 10,000 records).

If dimensionality was a problem, I would also try condensing the feature space down using dimensionality reduction techniques like Principle Component Analysis.

## How would you approach this challenge if the input dataset could not fit in your device memory?
As previous, I would look to employ a parallelisable/distributable model that allows out-of-core learning.  XGBoost could work for this, while also being the highest performing model tested.  (with more time I would have tried tuning each model, but so far have only considered baseline performance with default hyperparameters).

## What other considerations would need to be taken into account if this was real customer data?
If this was real customer data, I would need to consider the source of data and whether extra measures should be taken to protect privacy in the dataset

I would also want to consider factors like other demographic, geographic influences on the biases in this dataset.  This data looks like it is in dollars so probably covers US customers.  The classifier trained on this data might not generalise to other regions etc.
