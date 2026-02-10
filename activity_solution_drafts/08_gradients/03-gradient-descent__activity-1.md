---
notebook: 08_gradients/03-gradient-descent.ipynb
activity: 1
cell_index: 31
---

# Activity 1

This is Activity 1.

Source notebook: [`08_gradients/03-gradient-descent.ipynb`](../../08_gradients/03-gradient-descent.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown

Consider the following function.

$$f(\vec x) = (x_1 - 2)^2 + 2x_1 - (x_2-3)^2$$

1. Is $f(\vec x)$ a quadratic form?
1. Given an initial guess of $\vec{x}^{(0)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ and a step size of $\alpha = \frac{1}{3}$, perform **two** iterations of gradient descent. What is $\vec{x}^{(2)}$?
:::

## Draft Solution

\[
f(\vec x)=(x_1-2)^2+2x_1-(x_2-3)^2.
\]

1. Is it a quadratic form?
- **No.** A quadratic form has only terms \(\vec x^TQ\vec x\) (no linear or constant terms).
- Here there are linear and constant terms after expansion.

2. Gradient descent with \(\alpha=1/3\), \(\vec x^{(0)}=[0,0]^T\):

\[
\nabla f(x_1,x_2)=\begin{bmatrix}2x_1-2\\-2x_2+6\end{bmatrix}.
\]

At \(\vec x^{(0)}\):
\[
\nabla f(0,0)=\begin{bmatrix}-2\\6\end{bmatrix},
\quad
\vec x^{(1)}=\vec x^{(0)}-\alpha\nabla f(\vec x^{(0)})
=\begin{bmatrix}2/3\\-2\end{bmatrix}.
\]

At \(\vec x^{(1)}\):
\[
\nabla f\left(\frac23,-2\right)=\begin{bmatrix}-2/3\\10\end{bmatrix},
\]
\[
\vec x^{(2)}=\vec x^{(1)}-\frac13\nabla f(\vec x^{(1)})
=\begin{bmatrix}2/3\\-2\end{bmatrix}-\frac13\begin{bmatrix}-2/3\\10\end{bmatrix}
=\begin{bmatrix}8/9\\-16/3\end{bmatrix}.
\]

So
\[
\boxed{\vec x^{(2)}=\begin{bmatrix}8/9\\-16/3\end{bmatrix}}.
\]
