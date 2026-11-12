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

- **Gradient vector:** for $f: \mathbb{R}^d \to \mathbb{R}$,

  $$\nabla f(\vec x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_d} \end{bmatrix}$$

- **Steepest ascent:** $\nabla f(\vec x)$ points in the direction of steepest increase at $\vec x$, locally; it does not necessarily point toward the maximum.
- **Steepest descent:** the direction is $-\nabla f(\vec x)$.
- **Critical points** satisfy $\nabla f(\vec x) = \vec 0$. A critical point may be a minimum, maximum, or saddle.

### 8.2 Gradients of Matrix-Vector Operations

- **Gradient of a dot product:**

  $$\nabla(\vec a \cdot \vec x) = \vec a$$

- **Gradient of the squared norm:**

  $$\nabla\lVert \vec x \rVert^2 = 2\vec x$$

- **Gradient of a quadratic form:**

  $$\nabla(\vec x^TA\vec x) = (A + A^T)\vec x$$

- **Symmetric case:** if $A = A^T$,

  $$\nabla(\vec x^TA\vec x) = 2A\vec x$$

- **Chain rule:**

  $$\nabla h(g(\vec x)) = h'(g(\vec x))\,\nabla g(\vec x)$$

- **Gradient of the norm**, for $\vec x \ne \vec 0$:

  $$\nabla\lVert \vec x \rVert = \frac{\vec x}{\lVert \vec x \rVert}$$

- **Expanding MSE:** use $\vec y^TX\vec w = \vec w^TX^T\vec y$, since a scalar equals its transpose.

  $$R_\text{sq}(\vec w) = \frac{1}{n}\lVert\vec y - X\vec w\rVert^2$$

  $$= \frac{1}{n}\big(\vec y^T\vec y - 2\vec w^TX^T\vec y + \vec w^TX^TX\vec w\big)$$

- **Gradient of MSE:**

  $$\nabla R_\text{sq}(\vec w) = \frac{2}{n}\big(X^TX\vec w - X^T\vec y\big)$$

- Setting the gradient of MSE to $\vec 0$ gives the **same normal equations as before:**

  $$X^TX\vec w^* = X^T\vec y$$

### 8.3 Gradient Descent

- **Goal:** minimize a function $f: \mathbb{R}^d \to \mathbb{R}$, given its gradient. Use gradient descent especially when $\nabla f = \vec 0$ has no closed-form solution, which often happens in practice.
- Choose a **learning rate (step size)** $\alpha > 0$ and a starting point $\vec x^{(0)}$.
- **Gradient descent update:**

  $$\vec x^{(t+1)} = \vec x^{(t)} - \alpha\nabla f(\vec x^{(t)})$$

- **Stopping rule:** terminate when $\lVert\nabla f(\vec x^{(t)})\rVert \le 0.0001$, or some other small tolerance.
- **Starting point:** the result can depend on the start and can get stuck in a local minimum.
- **Learning rate (step size):** too large an $\alpha$ can cause bouncing; too small an $\alpha$ makes progress slow.
- A tiny gradient means the function is flat there, not necessarily optimal.

### 8.4 Gradient Descent for Empirical Risk Minimization

- Gradient descent is not necessary to minimize

  $$R_\text{sq}(\vec w) = \frac{1}{n}\lVert\vec y - X\vec w\rVert^2$$

  We've already found the minimizing $\vec w^*$ using two approaches: projections/orthogonality and a gradient.
- But, to illustrate, since

  $$\nabla R_\text{sq}(\vec w) = \frac{2}{n}(X^TX\vec w - X^T\vec y)$$

  $\vec w^*$ can be estimated using gradient descent:

  $$\vec w^{(t+1)} = \vec w^{(t)} - \alpha\frac{2}{n}(X^TX\vec w^{(t)} - X^T\vec y)$$

- Depending on the learning rate and size of the dataset, gradient descent *may* converge quicker than it takes to find a solution to the normal equations directly.

### 8.5 Convexity

- $f$ is **convex** if, for all $\vec x, \vec y$ and $t \in [0, 1]$,

  $$f\big((1 - t)\vec x + t\vec y\big) \le (1 - t)f(\vec x) + tf(\vec y)$$

- **Secant line interpretation:** all possible secant lines lie on or above the graph.

  :::{image} ../assets/summary-convexity.png
  :width: 460px
  :align: center
  :alt: The existing Section 8.5 figure of f(x) = x² below the secant joining x = -1 and x = 1.
  :::

- For convex $f$, **every local minimum and every critical point is a global minimum**.
- A global minimum need not exist; $e^x$ is an example.
- **Strict convexity:** replace $\le$ with $<$ when $\vec x \ne \vec y$ and $t \in (0, 1)$. A strictly convex function has at most one minimizer.
