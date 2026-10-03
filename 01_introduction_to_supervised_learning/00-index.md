---
numbering: false
---

# Chapter 1: Introduction to Supervised Learning

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 1.1 What is Machine Learning?

- **Supervised learning:** learn from labeled data to predict $y$ from features $x$.
  - **Classification** predicts a discrete category (e.g. true or false; cat, dog, or hamster; digit from 0 to 9).
  - **regression** predicts a real number (e.g. a commute time, a height, a weight).
- **Unsupervised learning:** finding structure in unlabeled data (e.g., clustering or dimensionality reduction).
- **Reinforcement learning:** training an agent to make decisions from rewards.
- **Overfitting:** a model that is too flexible may learn the quirks and noise of its training data and make worse predictions on new data than a simpler model.

### 1.2 Squared Loss and the Constant Model

- **Modeling recipe** (empirical risk minimization): choose a model, choose a loss, minimize average loss.
- Constant model: $h(x_i) = w$. Looks like a horizontal line; returns the same value for all $x_i$.
- $L$ is the loss for **one** point; $R$ is the **average** loss (**empirical risk**).
- Squared loss in general:
$$L_\text{sq}(y_i, h(x_i)) = (y_i - h(x_i))^2$$
- Mean squared error (average squared loss) for the constant model:
$$R_\text{sq}(w) = \frac{1}{n}\sum_{i=1}^n (y_i - w)^2$$
- To find the $w^*$ that minimizes the mean squared error, we take the derivative and set it to 0:
$$\frac{\text{d}}{\text{d}w}R_\text{sq}(w) = -\frac{2}{n}\sum_{i=1}^n (y_i - w) = 0 \implies w^* = \bar y$$
### 1.3 Absolute Loss
- Absolute loss in general:
$$L_\text{abs}(y_i, h(x_i)) = |y_i - h(x_i)|$$
- Mean absolute error (average absolute loss) for the constant model:
$$R_\text{abs}(w) = \frac{1}{n}\sum_{i=1}^n |y_i - w|$$
- $R_\text{abs}(w)$ is a piecewise linear function, with a bend at each data point, and is not differentiable. The slope of $R_\text{abs}(w)$ at any $w$ that is not a data point is:
$$\frac{\text{d}}{\text{d}w}R_\text{abs}(w) = \frac{(\#\text{ points left of } w) - (\#\text{ points right of } w)}{n}$$
- The **median** minimizes the mean absolute error (uniquely if $n$ is odd; for even $n$, so does any $w$ between the middle two).

### 1.4 Comparing Loss Functions

| Loss | $L(y_i, w)$ | $w^*$ | Always unique? | Robust to outliers? |
|---|---|---|---|---|
| squared | $(y_i - w)^2$ | mean | yes | no |
| absolute | $\lvert y_i - w \rvert$ | median | no | yes |
| $L_p$ as $p \to \infty$ | $\lvert y_i - w \rvert^p$ | midrange $\frac{\min + \max}{2}$ | yes | no |
| 0-1 | $0$ if $y_i = w$, else $1$ | mode | no | no |
- The mean is pulled toward outliers and the tail: **right-skewed** $\Rightarrow$ mean > median.
- Minimum risk measures **spread**: $$R_\text{sq}(\bar y) = \frac{1}{n}\sum (y_i - \bar y)^2 = \sigma_y^2$$ is the **variance** ($\sigma_y$ = its square root).
