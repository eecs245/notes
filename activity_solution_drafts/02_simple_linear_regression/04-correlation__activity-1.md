---
notebook: 02_simple_linear_regression/04-correlation.ipynb
activity: 1
cell_index: 13
---

# Activity 1

This is Activity 1.

Source notebook: [`02_simple_linear_regression/04-correlation.ipynb`](../../02_simple_linear_regression/04-correlation.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown

This activity is an old exam question, taken from an exam that used to allow calculators. Part of it also appears in Lab 3.

1. First, suppose we minimize mean squared error to fit a simple linear regression line that **uses the square footage of a house to predict its price**. The resulting line has an intercept of $w_0^*$ and a slope of $w_1^*$. In other words:

$$\text{predicted price}_i = w_0^* + w_1^* \cdot \text{square footage}_i$$

We're now interested in minimizing mean squared error to find a simple linear regression line that uses **price to predict square footage**. Suppose this new regression line has an intercept of $\beta_0^*$ and a slope of $\beta_1^*$.

What is $\beta_1^*$? Give your answer as an expression in terms of $n$, $r$, $w_0^*$, and/or $w_1^*$.

2. Given that:
- $n = 100$
- $r = 0.6$
- $w_0^* = 1000$
- $w_1^* = 250$|
- The average square footage of houses in the dataset is $2000$

What is $\beta_0^*$? Your answer should be a constant with no variables. (Once you're able to express $\beta_0^*$ in terms of constants only, you can stop simplifying your answer.)
:::

## Draft Solution

Let square footage be \(x\), price be \(y\).

For regression of \(y\) on \(x\):
\[
w_1^* = r\frac{\sigma_y}{\sigma_x}.
\]
For regression of \(x\) on \(y\):
\[
\beta_1^* = r\frac{\sigma_x}{\sigma_y}.
\]
Therefore
\[
w_1^*\beta_1^*=r^2
\quad\Longrightarrow\quad
\boxed{\beta_1^*=\frac{r^2}{w_1^*}}.
\]

Given \(r=0.6\), \(w_1^*=250\):
\[
\beta_1^*=\frac{0.36}{250}=0.00144.
\]

Now use the fact that each OLS line passes through \((\bar x,\bar y)\):
\[
\bar y=w_0^*+w_1^*\bar x=1000+250(2000)=501000.
\]
For the reverse regression,
\[
\beta_0^*=\bar x-\beta_1^*\bar y
=2000-0.00144(501000)
=1278.56.
\]

So
\[
\boxed{\beta_0^*=1278.56}.
\]
