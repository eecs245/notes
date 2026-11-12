---
numbering: false
---

# Chapter 7: Regression using Linear Algebra

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 7.1 Regression using Linear Algebra

- **Simple linear regression model:**

  $$h(x_i) = w_0 + w_1x_i$$

- **Observation vector:** $\vec y \in \mathbb{R}^n$ contains the observed outcomes.
- **Design matrix:** $X$ has shape $n \times 2$. The 1s give the intercept.

  $$X = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}$$
- **Parameter vector:**

  $$\vec w = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}$$

- **Prediction vector:** $X\vec w$.
- **Error vector:**

  $$\vec e = \vec y - X\vec w$$

- **Mean squared error:**

  $$R_\text{sq}(\vec w) = \frac{1}{n}\sum_{i=1}^n \big(y_i - (w_0 + w_1x_i)\big)^2$$

- **Vector form of MSE:**

  $$R_\text{sq}(\vec w) = \frac{1}{n}\lVert \vec y - X\vec w \rVert^2$$

- **Geometric interpretation:** minimizing MSE finds the vector in $\text{colsp}(X)$ closest to $\vec y$, the orthogonal projection $\vec p = X\vec w^*$.
- **Normal equations:**

  $$X^TX\vec w^* = X^T\vec y$$

- **Equivalent orthogonality condition:**

  $$X^T(\vec y - X\vec w^*) = \vec 0$$

- **Optimal parameters**, with linearly independent columns:

  $$\vec w^* = (X^TX)^{-1}X^T\vec y$$

- **Projection**, with linearly independent columns:

  $$\vec p = X(X^TX)^{-1}X^T\vec y$$

- **Minimum MSE:**

  $$\frac{1}{n}\lVert \vec y - \vec p \rVert^2$$

- $X$ is usually not square, so there is no $X^{-1}$. If $X$ is square and invertible, $\vec w^* = X^{-1}\vec y$ gives an exact fit.
- The columns of the simple linear regression design matrix are linearly independent unless every $x_i$ is equal.
- The solution

  $$\vec w^* = \begin{bmatrix} w_0^* \\ w_1^* \end{bmatrix} = (X^TX)^{-1}X^T\vec y$$

  matches the formulas from Chapter 2:

  $$w_1^* = r\frac{\sigma_y}{\sigma_x}$$

  $$w_0^* = \bar y - w_1^*\bar x$$

- **Errors sum to zero:** with an intercept, the 1s column gives

  $$\vec 1 \cdot (\vec y - X\vec w^*) = 0$$

- **Data-space picture:** the data and fitted line lie in $\mathbb{R}^2$.
- **Projection picture:** $\vec y$ is projected onto $\text{colsp}(X)$ in $\mathbb{R}^n$.
- **Loss-surface picture:** the MSE surface over $(w_0, w_1)$ lies in $\mathbb{R}^3$ and is lowest at $\vec w^*$.

### 7.2 Multiple Linear Regression

- $x_i^{(j)}$ is feature $j$ of individual $i$. With $d$ features and an intercept, there are $d + 1$ parameters.
- **Augmented feature vector:**

  $$\text{Aug}(\vec x_i) = \begin{bmatrix} 1 \\ x_i^{(1)} \\ \vdots \\ x_i^{(d)} \end{bmatrix}$$

- **Design matrix:** $X$ has shape $n \times (d + 1)$, with rows $\text{Aug}(\vec x_i)^T$.
- **Multiple linear regression model:**

  $$h(\vec x_i) = w_0 + w_1x_i^{(1)} + \cdots + w_dx_i^{(d)}$$

- **Dot-product form:**

  $$h(\vec x_i) = \vec w \cdot \text{Aug}(\vec x_i)$$

- The same normal equations apply. The optimal $\vec w^*$ is unique iff the $d + 1$ columns are linearly independent; otherwise there are infinitely many solutions, all with the same predictions and MSE.
- **One-hot encoding:** $k$ categories become $k$ binary columns. Codes like Mon = 1, Tue = 2 would impose a meaningless order.
- **Dependence with an intercept:** the $k$ one-hot columns sum to the 1s column, so the columns are dependent and $\vec w^*$ is not unique.
- **Dropping one category:** the column space, predictions, and MSE do not change. The remaining $k - 1$ one-hot coefficients are shifts from the dropped (baseline) category.
- **Linear in the parameters:** linear regression fits functions that are linear combinations of features, such as

  $$h(x_i) = w_0 + w_1x_i + w_2x_i^2$$

  This is a plane in feature space (where the axes are $x$, $x^2$, and $h(x)$), but a curve in $(x, h(x))$ space.
- A parameter inside a nonlinear function, such as $\sin(w_2x)$, cannot be fit this way.
- Adding features never increases **training MSE**, since the column space never shrinks.
- **Overfitting:** test MSE can rise as features are added. Fit candidate models on training data and compare them on test data.
- **scikit-learn:** `LinearRegression().fit(features, y)` adds the intercept itself. `intercept_` is $w_0^*$; `coef_` contains the remaining coefficients.
- If $X$ already has the 1s column, pass `fit_intercept=False`.
- `predict` needs 2D input.
