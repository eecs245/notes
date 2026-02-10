---
notebook: 04_linear_independence/04-lines-planes-hyperplanes.ipynb
activity: 7
cell_index: 20
---

# Activity 7

This is Activity 7.

Source notebook: [`04_linear_independence/04-lines-planes-hyperplanes.ipynb`](../../04_linear_independence/04-lines-planes-hyperplanes.ipynb)

## Activity Text

:::{tip} Activity 7
:class: dropdown
1. Find the equation, in standard form, of the plane spanned by $\begin{bmatrix} 3 \\ 2 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix}$. Why did you not need to compute the cross product?
1. Find the equation, in standard form, of the plane spanned by $\begin{bmatrix} 3 \\ 2 \\ 4 \end{bmatrix}$ and $\begin{bmatrix} -1 \\ 1 \\ 2 \end{bmatrix}$.
:::

## Draft Solution

1. For
\[
\vec a=\begin{bmatrix}3\\2\\0\end{bmatrix},
\quad
\vec b=\begin{bmatrix}-1\\1\\0\end{bmatrix},
\]
both vectors have \(z=0\), so their span is the \(xy\)-plane:
\[
\boxed{z=0}.
\]
No cross product needed because this is immediate from coordinates.

2. For
\[
\vec a=\begin{bmatrix}3\\2\\4\end{bmatrix},
\quad
\vec b=\begin{bmatrix}-1\\1\\2\end{bmatrix},
\]
a normal vector is
\[
\vec a\times\vec b=
\begin{bmatrix}
0\\-10\\5
\end{bmatrix}
\sim
\begin{bmatrix}0\\-2\\1\end{bmatrix}.
\]
So plane equation in standard form is
\[
0x-2y+z=0
\quad\Longleftrightarrow\quad
\boxed{z=2y}.
\]
