---
notebook: 02_simple_linear_regression/02-partial-derivatives.ipynb
activity: 2
cell_index: 11
---

# Activity 2

This is Activity 2.

Source notebook: [`02_simple_linear_regression/02-partial-derivatives.ipynb`](../../02_simple_linear_regression/02-partial-derivatives.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown

Find the values of $x_1$ and $x_2$ that minimize the function:

$$g(x_1, x_2) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2$$

Here, we've used $x_1$ and $x_2$ to denote the two input variables, rather than $x$ and $y$.
:::

## Draft Solution

\[
g(x_1,x_2)=100(x_2-x_1^2)^2+(1-x_1)^2.
\]
This is a sum of squares, so \(g\ge 0\). The minimum possible value is 0, achieved exactly when both squared terms are 0:
\[
x_2-x_1^2=0,\quad 1-x_1=0.
\]
Hence
\[
x_1=1,\quad x_2=1.
\]
So the minimizer is
\[
\boxed{(x_1^*,x_2^*)=(1,1)},
\]
with minimum value \(g(1,1)=0\).
