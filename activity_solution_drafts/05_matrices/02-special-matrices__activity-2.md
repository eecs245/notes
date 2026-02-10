---
notebook: 05_matrices/02-special-matrices.ipynb
activity: 2
cell_index: 4
---

# Activity 2

This is Activity 2.

Source notebook: [`05_matrices/02-special-matrices.ipynb`](../../05_matrices/02-special-matrices.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown

Let $X = \begin{bmatrix} 1 & -2 \\ -1 & 3 \\ 2 & 0 \\ 0 & -1 \\ 3 & 2 \end{bmatrix}$.

1. Compute $X^TX$. 
2. Then, compute the transpose of $X^TX$. What do you notice? ($X^TX$ is called a _symmetric_ matrix.)
3. Compute $X^TX + \frac{1}{2} I$. We'll use matrices of the form $X^TX + \lambda I$ in Chapter 5.
:::

## Draft Solution

\[
X=
\begin{bmatrix}
1&-2\\
-1&3\\
2&0\\
0&-1\\
3&2
\end{bmatrix}.
\]

1.
\[
X^TX=
\begin{bmatrix}
15&1\\
1&18
\end{bmatrix}.
\]

2.
\[
(X^TX)^T=
\begin{bmatrix}
15&1\\
1&18
\end{bmatrix}=X^TX.
\]
So \(X^TX\) is symmetric.

3.
\[
X^TX+\frac12 I=
\begin{bmatrix}
15.5&1\\
1&18.5
\end{bmatrix}
=
\begin{bmatrix}
31/2&1\\
1&37/2
\end{bmatrix}.
\]
