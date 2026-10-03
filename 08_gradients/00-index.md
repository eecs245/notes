---
numbering: false
---

# Chapter 8: Gradients

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 8.1 The Gradient Vector

$$\nabla f(\vec x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \cdots & \frac{\partial f}{\partial x_d} \end{bmatrix}^T \qquad (f: \mathbb{R}^d \to \mathbb{R})$$

- $\nabla f(\vec x)$ points in the direction of **steepest ascent** at $\vec x$ (locally; not necessarily toward the maximum); $-\nabla f$ is steepest descent. **Critical points** are where $\nabla f = \vec 0$: a minimum, maximum, or saddle.

### 8.2 Gradients of Matrix-Vector Operations

$$\nabla(\vec a \cdot \vec x) = \vec a \qquad\quad \nabla \lVert \vec x \rVert^2 = 2\vec x \qquad\quad \nabla(\vec x^TA\vec x) = (A + A^T)\vec x = 2A\vec x \;\text{ if } A \text{ is symmetric}$$

- **Chain rule:** $\nabla h(g(\vec x)) = h'(g(\vec x))\,\nabla g(\vec x)$. E.g. $\nabla \lVert \vec x \rVert = \frac{\vec x}{\lVert \vec x \rVert}$ ($\vec x \ne \vec 0$), and $\nabla \log \sum_j e^{x_j} = \text{softmax}(\vec x)$, with entries $\frac{e^{x_i}}{\sum_j e^{x_j}}$ (summing to 1).
- For symmetric $A$: $\nabla(\vec x^TA\vec x + \vec b \cdot \vec x + c) = 2A\vec x + \vec b$, so the critical point is $\vec x^* = -\frac{1}{2}A^{-1}\vec b$ ($A$ invertible).
- **MSE:** expand, using $\vec y^TX\vec w = \vec w^TX^T\vec y$ (a scalar equals its transpose), then apply the rules:

$$R_\text{sq}(\vec w) = \frac{1}{n}\big(\vec y^T\vec y - 2\vec w^TX^T\vec y + \vec w^TX^TX\vec w\big) \qquad \nabla R_\text{sq}(\vec w) = \frac{2}{n}\big(X^TX\vec w - X^T\vec y\big)$$

- Setting $\nabla R_\text{sq}(\vec w) = \vec 0$ gives the normal equations $X^TX\vec w^* = X^T\vec y$ again.

### 8.3 Gradient Descent

- Choose a **learning rate** $\alpha > 0$ and a start $\vec x^{(0)}$; repeat until $\lVert \nabla f(\vec x^{(t)}) \rVert$ is below a tolerance:

$$\vec x^{(t+1)} = \vec x^{(t)} - \alpha \nabla f(\vec x^{(t)})$$

- Use it when $\nabla f = \vec 0$ has no closed-form solution (e.g. logistic regression).
- The result depends on the start and $\alpha$: it can get stuck in a **local** minimum; too large an $\alpha$ bounces, too small is slow. A tiny gradient means flat, not necessarily optimal.

### 8.4 Gradient Descent for Empirical Risk Minimization

- For MSE, $\vec w^{(t+1)} = \vec w^{(t)} - \alpha\frac{2}{n}(X^TX\vec w^{(t)} - X^T\vec y)$.

### 8.5 Convexity

$$f\big((1 - t)\vec x + t\vec y\big) \le (1 - t)f(\vec x) + tf(\vec y) \quad \text{for all } \vec x, \vec y \text{ and } t \in [0, 1] \qquad \text{(convex)}$$

- Secant lines lie on or above the graph. A differentiable $f$ is convex iff it lies above every tangent hyperplane: $f(\vec y) \ge f(\vec x) + \nabla f(\vec x)^T(\vec y - \vec x)$. So for convex $f$, **every local minimum and every critical point is a global minimum** (one needn't exist: $e^x$).
- **Strictly convex** ($<$ for $\vec x \ne \vec y$, $t \in (0, 1)$): at most one minimizer.
- **Hessian** $H_f$: the matrix of second partials $\frac{\partial^2 f}{\partial x_i \partial x_j}$. $f$ is convex iff $H_f$ is **PSD** ($\vec v^TH_f\vec v \ge 0$ for all $\vec v$) everywhere; **PD** ($> 0$ for all $\vec v \ne \vec 0$) everywhere implies strictly convex.
- MSE's Hessian is $\frac{2}{n}X^TX$, and $\vec v^TX^TX\vec v = \lVert X\vec v \rVert^2 \ge 0$. So MSE is always convex, and strictly convex (unique $\vec w^*$) iff $X$'s columns are independent.
