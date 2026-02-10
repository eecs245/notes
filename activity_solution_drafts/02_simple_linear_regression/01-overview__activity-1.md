---
notebook: 02_simple_linear_regression/01-overview.ipynb
activity: 1
cell_index: 4
---

# Activity 1

This is Activity 1.

Source notebook: [`02_simple_linear_regression/01-overview.ipynb`](../../02_simple_linear_regression/01-overview.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown

Consider a dataset with two points, $(3, 5)$ and $(15, 53)$. What are the optimal parameters, $w_0^*$ and $w_1^*$, for the line $h(x_i) = w_0 + w_1 x_i$ that minimizes mean squared error for this dataset?
:::

## Draft Solution

With two points, the best-fit line under MSE can pass through both exactly (zero training error), so solve for the unique line through \((3,5)\) and \((15,53)\).

\[
w_1^*=\frac{53-5}{15-3}=\frac{48}{12}=4,
\]
\[
w_0^*=5-4\cdot 3=-7.
\]

So
\[
\boxed{w_0^*=-7,\quad w_1^*=4}.
\]
