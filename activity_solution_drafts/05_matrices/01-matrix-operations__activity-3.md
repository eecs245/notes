---
notebook: 05_matrices/01-matrix-operations.ipynb
activity: 3
cell_index: 8
---

# Activity 3

This is Activity 3.

Source notebook: [`05_matrices/01-matrix-operations.ipynb`](../../05_matrices/01-matrix-operations.ipynb)

## Activity Text

:::{tip} Activity 3
:class: dropdown

Consider the matrix $M$ defined below.

$$M = \begin{bmatrix} 2 & -1 & 3 & 0 & 4 \\ 1 & 5 & -2 & 1 & 0 \end{bmatrix}$$

In each of the following parts, write out $\vec u$ concretely, compute $M \vec u$, and explain the result in English.

1. A vector whose second component is 1, and whose other components are 0.
2. A vector containing all 1s.
3. A vector containing all $\frac{1}{5}$s.
4. A vector whose components sum to 1, whose first component is $\frac{3}{5}$, and whose other components are all equal to one another.
:::

## Draft Solution

\[
M=\begin{bmatrix}2&-1&3&0&4\\1&5&-2&1&0\end{bmatrix}.
\]

1. "Second component is 1, others 0":
\[
\vec u=\begin{bmatrix}0\\1\\0\\0\\0\end{bmatrix},
\quad
M\vec u=
\begin{bmatrix}-1\\5\end{bmatrix}.
\]
This returns column 2 of \(M\).

2. All ones:
\[
\vec u=\begin{bmatrix}1\\1\\1\\1\\1\end{bmatrix},
\quad
M\vec u=
\begin{bmatrix}8\\5\end{bmatrix}.
\]
This is the sum of all columns of \(M\).

3. All \(1/5\):
\[
\vec u=\frac15\begin{bmatrix}1\\1\\1\\1\\1\end{bmatrix},
\quad
M\vec u=\frac15\begin{bmatrix}8\\5\end{bmatrix}
=\begin{bmatrix}8/5\\1\end{bmatrix}.
\]
This scales the previous result by \(1/5\).

4. Components sum to 1, first is \(3/5\), others equal:
\[
\vec u=\begin{bmatrix}3/5\\1/10\\1/10\\1/10\\1/10\end{bmatrix}
\]
(since \(3/5+4a=1\Rightarrow a=1/10\)). Then
\[
M\vec u=
\begin{bmatrix}9/5\\1\end{bmatrix}.
\]
Interpretation: this is a weighted linear combination of columns of \(M\) with weights summing to 1.
