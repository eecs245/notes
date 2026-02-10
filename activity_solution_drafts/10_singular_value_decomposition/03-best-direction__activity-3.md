---
notebook: 10_singular_value_decomposition/03-best-direction.ipynb
activity: 3
cell_index: 23
---

# Activity 3

This is Activity 3.

Source notebook: [`10_singular_value_decomposition/03-best-direction.ipynb`](../../10_singular_value_decomposition/03-best-direction.ipynb)

## Activity Text

:::{tip} Activity 3
:class: dropdown
Suppose $x, y \in \mathbb{R}$. What are the largest and smallest possible values of

$$f(x, y) = \frac{2x^2 + 12xy + 7y^2}{x^2 + y^2}$$
:::

## Draft Solution

Write
\[
f(x,y)=\frac{\begin{bmatrix}x&y\end{bmatrix}
\begin{bmatrix}2&6\\6&7\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}}{x^2+y^2}.
\]
This is a Rayleigh quotient of the symmetric matrix
\[
A=\begin{bmatrix}2&6\\6&7\end{bmatrix}.
\]
Its minimum and maximum values are the smallest and largest eigenvalues of \(A\).

Compute eigenvalues:
\[
\det(A-\lambda I)=(2-\lambda)(7-\lambda)-36
=\lambda^2-9\lambda-22=0.
\]
So
\[
\lambda=\frac{9\pm\sqrt{81+88}}{2}=\frac{9\pm13}{2}.
\]
Hence eigenvalues are \(-2\) and \(11\).

Therefore
\[
\boxed{\min f=-2,\quad \max f=11}.
\]
