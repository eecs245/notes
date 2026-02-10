---
notebook: 00_math_foundations/01-summation.ipynb
activity: 4
cell_index: 3
---

# Activity 4

This is Activity 4.

Source notebook: [`00_math_foundations/01-summation.ipynb`](../../00_math_foundations/01-summation.ipynb)

## Activity Text

:::{tip} Activity 4
:class: dropdown

Suppose the dataset $x_1, x_2, \ldots, x_n$ has mean $\bar{x}$ and variance $\sigma^2$.

Find the mean and variance of the dataset $-4x_1 + 3, -4x_2 + 3, \ldots, -4x_n + 3$, and justify your answer rigorously using the definitions of mean and variance.

:::

## Draft Solution

Let \(y_i=-4x_i+3\).

Mean:
\[
\bar y=\frac1n\sum_{i=1}^n(-4x_i+3)=-4\left(\frac1n\sum_{i=1}^n x_i\right)+3=-4\bar x+3.
\]

Variance (using population variance definition):
\[
\sigma_y^2=\frac1n\sum_{i=1}^n(y_i-\bar y)^2
=\frac1n\sum_{i=1}^n\left((-4x_i+3)-(-4\bar x+3)\right)^2
\]
\[
=\frac1n\sum_{i=1}^n\left(-4(x_i-\bar x)\right)^2
=16\cdot \frac1n\sum_{i=1}^n(x_i-\bar x)^2
=16\sigma^2.
\]

So the transformed dataset has mean \(\boxed{-4\bar x+3}\) and variance \(\boxed{16\sigma^2}\).
