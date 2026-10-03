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

- For $h(x_i) = w_0 + w_1 x_i$: the observation vector is $\vec y \in \mathbb{R}^n$, the $n \times 2$ **design matrix** $X$ has rows $\begin{bmatrix} 1 & x_i \end{bmatrix}$ (the 1s give the intercept), $\vec w = \begin{bmatrix} w_0 & w_1 \end{bmatrix}^T$, the predictions are $X\vec w$, and the errors are $\vec e = \vec y - X\vec w$:

$$R_\text{sq}(\vec w) = \frac{1}{n}\sum_{i=1}^n \big(y_i - (w_0 + w_1 x_i)\big)^2 = \frac{1}{n}\lVert \vec y - X\vec w \rVert^2$$

- Minimizing MSE means finding the vector in $\text{colsp}(X)$ closest to $\vec y$: the **orthogonal projection** $\vec p = X\vec w^*$, where

$$X^TX\vec w^* = X^T\vec y \iff X^T(\vec y - X\vec w^*) = \vec 0 \qquad\quad \vec w^* = (X^TX)^{-1}X^T\vec y \;\; \text{(independent columns)}$$

- With independent columns, $\vec p = X(X^TX)^{-1}X^T\vec y$; always $n \cdot \text{MSE} = \lVert \vec y - X\vec w^* \rVert^2$. $X$ is usually not square, so there's no $X^{-1}$ (a square, invertible $X$ would give $\vec w^* = X^{-1}\vec y$, an exact fit).
- The columns are independent unless every $x_i$ is equal, and the answer matches Chapter 2: $w_1^* = r\frac{\sigma_y}{\sigma_x}$, $w_0^* = \bar y - w_1^* \bar x$. By hand: $X^TX = \begin{bmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{bmatrix}$, $X^T\vec y = \begin{bmatrix} \sum y_i \\ \sum x_i y_i \end{bmatrix}$.
- With an intercept, the optimal errors sum to 0: the 1s column gives $\vec 1 \cdot (\vec y - X\vec w^*) = 0$.
- Three pictures: the data and line in $\mathbb{R}^2$; $\vec y$ projected onto $\text{colsp}(X)$ in $\mathbb{R}^n$; the MSE surface over $(w_0, w_1)$ in $\mathbb{R}^3$, lowest at $\vec w^*$.

### 7.2 Multiple Linear Regression

- $x_i^{(j)}$ is feature $j$ of individual $i$, and $d$ features give $d + 1$ parameters. With the **augmented feature vector** $\text{Aug}(\vec x_i) = \begin{bmatrix} 1 & x_i^{(1)} & \cdots & x_i^{(d)} \end{bmatrix}^T$, the design matrix $X$ is $n \times (d + 1)$ with rows $\text{Aug}(\vec x_i)^T$, and

$$h(\vec x_i) = w_0 + w_1 x_i^{(1)} + \cdots + w_d x_i^{(d)} = \vec w \cdot \text{Aug}(\vec x_i)$$

- Same normal equations: $\vec w^*$ is unique iff the $d + 1$ columns are independent; otherwise there are infinitely many, all with the same $\vec p$ and MSE. Predict with $\vec w^* \cdot \text{Aug}(\vec x)$ (an association, not a cause).
- **One-hot encoding:** $k$ categories become $k$ binary columns (codes like Mon = 1, Tue = 2 would impose a meaningless order). With an intercept the $k$ columns sum to the 1s column (so $X$ isn't full rank and $\vec w^*$ isn't unique); drop one: $\text{colsp}(X)$, the predictions, and the MSE don't change. The $k - 1$ one-hot coefficients are shifts from the dropped (baseline) category.
- **Linear in the parameters:** linear regression fits any $h = \vec w \cdot (\text{features})$ whose features are fixed functions of the inputs, e.g. $w_0 + w_1 x + w_2 x^2$ (a plane in feature space, a curve against $x$). A parameter inside a nonlinear function, like $\sin(w_2 x)$, can't be fit this way.
- Adding features never increases **training** MSE ($\text{colsp}(X)$ never shrinks), but test MSE can rise (**overfitting**). Fit candidate models on training data and compare them on test data.
- `LinearRegression().fit(features, y)` adds the intercept itself (`intercept_` is $w_0^*$, `coef_` is $[w_1^*, \dots]$); if $X$ already has the 1s column, pass `fit_intercept=False` (then `coef_` is all of $\vec w^*$, starting with $w_0^*$). `predict` needs 2D input. Or solve directly: `np.linalg.solve(X.T @ X, X.T @ y)`.
