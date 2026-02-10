---
notebook: 05_matrices/01-matrix-operations.ipynb
activity: 2
cell_index: 7
---

# Activity 2

This is Activity 2.

Source notebook: [`05_matrices/01-matrix-operations.ipynb`](../../05_matrices/01-matrix-operations.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown

We'll try not to bore you with mundane calculations in the future, but it's important to perform matrix-vector multiplication by hand a few times to understand how it works. 

In each part, perform the matrix-vector multiplication by hand or state that it cannot be done.

1. $$\begin{bmatrix} 7 & 8 & -1 \\ 0 & 1 & 2 \\ 9 & 0 & \frac{1}{2} \end{bmatrix} \begin{bmatrix} 5 \\ 4 \\ 8 \end{bmatrix}$$
1. $$\begin{bmatrix} 7 & 8 & -1 \\ 0 & 1 & 2 \\ \end{bmatrix} \begin{bmatrix} 5 \\ 4 \end{bmatrix}$$
1. $$\begin{bmatrix} 4 & 2 & 3 \\ 1 & 0 & 1 \\ 5 & 4 & 3 \\ 0 & 0 & 1 \\ \end{bmatrix} \begin{bmatrix} 5 \\ 4 \\ 8 \end{bmatrix}$$
1. $$\begin{bmatrix} 4 & 2 & 3 \\ 1 & 0 & 1 \\ 5 & 4 & 3 \\ 0 & 0 & 1 \\ \end{bmatrix} \begin{bmatrix} 7 & 8 & -1 \\ 0 & 1 & 2 \\ 9 & 0 & \frac{1}{2} \end{bmatrix} \begin{bmatrix} 5 \\ 4 \\ 8 \end{bmatrix}$$
(While we haven't yet looked at how to compute the product of two matrices, you can still answer this just using what you know about matrix-vector multiplication.)

In the cell below, use `numpy` to verify your answers. You'll need to define the matrices and vectors as `numpy` arrays, and use the `@` operator to perform the matrix-vector multiplication.

<iframe
src="https://jupyterlite.github.io/demo/repl/index.html?kernel=python&code=import numpy as np&code=A = np.array([[7, 8, -1], [0, 1, 2], [9, 0, 1/2]])&code=B = np.array([[5], [4], [8]])&code=A @ B"
width="100%"
height="600px"
></iframe>

:::

## Draft Solution

1.
\[
\begin{bmatrix}7&8&-1\\0&1&2\\9&0&1/2\end{bmatrix}
\begin{bmatrix}5\\4\\8\end{bmatrix}
=
\begin{bmatrix}59\\20\\49\end{bmatrix}.
\]

2.
\[
\begin{bmatrix}7&8&-1\\0&1&2\end{bmatrix}
\begin{bmatrix}5\\4\end{bmatrix}
\]
is **not defined** (inner dimensions 3 and 2 do not match).

3.
\[
\begin{bmatrix}4&2&3\\1&0&1\\5&4&3\\0&0&1\end{bmatrix}
\begin{bmatrix}5\\4\\8\end{bmatrix}
=
\begin{bmatrix}52\\13\\65\\8\end{bmatrix}.
\]

4. The full product is defined. Compute middle-times-right first:
\[
\begin{bmatrix}7&8&-1\\0&1&2\\9&0&1/2\end{bmatrix}
\begin{bmatrix}5\\4\\8\end{bmatrix}
=
\begin{bmatrix}59\\20\\49\end{bmatrix}.
\]
Then
\[
\begin{bmatrix}4&2&3\\1&0&1\\5&4&3\\0&0&1\end{bmatrix}
\begin{bmatrix}59\\20\\49\end{bmatrix}
=
\begin{bmatrix}423\\108\\522\\49\end{bmatrix}.
\]
