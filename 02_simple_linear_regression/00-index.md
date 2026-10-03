---
numbering: false
---

# Chapter 2: Simple Linear Regression

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 2.1 Overview
- **Simple linear regression model:** $$h(x_i) = w_0 + w_1 x_i$$ $w_0$ is the intercept; $w_1$ is the slope.
- Mean squared error for the simple linear regression model is a bowl-shaped function of $w_0$ and $w_1$:
$$R_\text{sq}(w_0, w_1) = \frac{1}{n}\sum_{i=1}^n \big(y_i - (w_0 + w_1 x_i)\big)^2$$

### 2.2 Partial Derivatives

- Suppose $f(x, y, z, ...)$ is a function of multiple input variables. $\frac{\partial f}{\partial x}$ – the **partial derivative** of $f$ with respect to $x$ – is the derivative of $f$ with respect to $x$, treating all other inputs as constants. It's still a function of **all** inputs.
- Critical points: all partial derivatives are 0 at once. E.g. $f(x, y) = \frac{x^2 + y^2}{9}$ has partials $\frac{2x}{9}, \frac{2y}{9}$; minimum at $(0, 0)$.

### 2.3 Finding Optimal Parameters
- The partial derivatives of $R_\text{sq}(w_0, w_1)$ are:

$$\frac{\partial R_\text{sq}}{\partial w_0} = -\frac{2}{n}\sum_{i=1}^n \big(y_i - (w_0 + w_1 x_i)\big)$$
$$\frac{\partial R_\text{sq}}{\partial w_1} = -\frac{2}{n}\sum_{i=1}^n x_i \big(y_i - (w_0 + w_1 x_i)\big)$$

- Setting both partials to 0 and solving for $w_0$ and $w_1$ gives:
$$w_1^* = \frac{\sum_{i=1}^n (x_i - \bar x)(y_i - \bar y)}{\sum_{i=1}^n (x_i - \bar x)^2} \qquad\quad w_0^* = \bar y - w_1^* \bar x$$

- Since $\sum_{i=1}^n (x_i - \bar x) = 0$, also $w_1^* = \frac{\sum_{i=1}^n (x_i - \bar x)\, y_i}{\sum_{i=1}^n (x_i - \bar x)^2}$, and several other useful forms.
- The errors of the fit line satisfy $\sum_{i=1}^n (y_i - h(x_i)) = 0$.
- The fit line passes through $(\bar x, \bar y)$.

### 2.4 Correlation

- Correlation coefficient:
$$r = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \bar x}{\sigma_x}\right)\left(\frac{y_i - \bar y}{\sigma_y}\right)$$where $$\sigma_x = \sqrt{\frac{1}{n}\sum_{i=1}^n (x_i - \bar x)^2}$$

- $r$ is the mean product of z-scores. $-1 \le r \le 1$; $|r|$ is the strength and the sign the direction; it is unitless, and $r(x, y) = r(y, x)$. It measures **linear** association only: $r \approx 0$ can hide a strong curve.
- $x_i \to a x_i + b$, $y_i \to c y_i + d$ ($a, c \ne 0$) leaves $|r|$ unchanged; $r$ flips sign if exactly one of $a, c$ is negative.
- The optimal slope can be written in terms of $r$:
$$w_1^* = r\,\frac{\sigma_y}{\sigma_x} \qquad\quad w_0^* = \bar y - w_1^* \bar x$$

- So the line depends only on $\bar x, \bar y, \sigma_x, \sigma_y, r$.

### 2.5 Least Squares
- Summary of models so far:

| Model | Optimal parameters | Minimum MSE |
|---|---|---|
| $h(x_i) = w$ | $w^* = \bar y$ | $\sigma_y^2$ |
| $h(x_i) = w_0 + w_1 x_i$ | $w_1^* = r\frac{\sigma_y}{\sigma_x}$, $\; w_0^* = \bar y - w_1^* \bar x$ | $\sigma_y^2 (1 - r^2) \le \sigma_y^2$ |

- **Least squares** is the general process of finding optimal parameters by minimizing the mean squared error, or equivalently the sum of squared errors (same minimizer).