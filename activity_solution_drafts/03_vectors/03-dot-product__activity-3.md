---
notebook: 03_vectors/03-dot-product.ipynb
activity: 3
cell_index: 11
---

# Activity 3

This is Activity 3.

Source notebook: [`03_vectors/03-dot-product.ipynb`](../../03_vectors/03-dot-product.ipynb)

## Activity Text

:::{tip} Activity 3
:class: dropdown

**Activity 3.1**

Find a value of $k$ such that the vectors $\vec u = \begin{bmatrix} 9 \\ -2 \\ 1 \end{bmatrix}$ and $\vec v = \begin{bmatrix} 1 \\ k \\ 3 \end{bmatrix}$ are orthogonal. 

Is this value of $k$ unique?

**Activity 3.2**

Find a vector that is orthogonal to **both** $\vec u = \begin{bmatrix} 1 \\ -2 \\ 4 \end{bmatrix}$ and $\vec v = \begin{bmatrix} 3 \\ -1 \\ 9 \end{bmatrix}$. In $\mathbb{R}^3$, what does this new vector look like, relative to $\vec u$ and $\vec v$?
:::

## Draft Solution

**Activity 3.1**

Orthogonality requires dot product 0:
\[
\begin{bmatrix}9\\-2\\1\end{bmatrix}\cdot\begin{bmatrix}1\\k\\3\end{bmatrix}=9-2k+3=12-2k=0
\Rightarrow k=6.
\]
This value is unique (linear equation with one solution).

**Activity 3.2**

A vector orthogonal to both can be the cross product:
\[
\begin{bmatrix}1\\-2\\4\end{bmatrix}\times\begin{bmatrix}3\\-1\\9\end{bmatrix}
=\begin{bmatrix}-14\\3\\5\end{bmatrix}.
\]
Any nonzero scalar multiple also works.

Geometrically in \(\mathbb R^3\), this vector is normal (perpendicular) to the plane spanned by the two given vectors.
