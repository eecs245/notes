---
notebook: 07_regression_using_linear_algebra/02-multiple-linear-regression.ipynb
activity: 1
cell_index: 29
---

# Activity 1

This is Activity 1.

Source notebook: [`07_regression_using_linear_algebra/02-multiple-linear-regression.ipynb`](../../07_regression_using_linear_algebra/02-multiple-linear-regression.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown
Recall, **training data** is the data that we use to fit/train/create the model. **Test data** refers to any other data that we might use to evaluate the model's performance.

1. As we add more features, the mean squared error of a model's predictions **on the training data** will **never decrease**. Why?
1. Could a model's mean squared error **on test data** increase as we add more features? When would it?
:::

## Draft Solution

Assumption: the first sentence likely has a typo. For training error, adding features means MSE can only stay the same or go down (it should say "never increase").

1. Why training MSE cannot increase:
- Adding features enlarges the model class.
- The old model is still representable by setting new-feature coefficients to 0.
- So the new optimum cannot be worse on training data.

2. Could test MSE increase?
- Yes. This is overfitting.
- With many features (especially noisy/irrelevant ones) relative to sample size, variance increases and the model can fit training noise.
- Then generalization degrades, so test MSE can rise.
