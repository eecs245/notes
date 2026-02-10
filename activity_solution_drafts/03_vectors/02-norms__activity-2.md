---
notebook: 03_vectors/02-norms.ipynb
activity: 2
cell_index: 3
---

# Activity 2

This is Activity 2.

Source notebook: [`03_vectors/02-norms.ipynb`](../../03_vectors/02-norms.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown

The triangle inequality says that:

$$\lVert \vec u + \vec v \rVert \leq \lVert \vec u \rVert + \lVert \vec v \rVert$$

In the example above, we had $\sqrt{74} < \sqrt{10} + \sqrt{52}$. In other words, there was **strict** inequality.

Find a pair of vectors $\vec u, \vec v$ (say, in $\mathbb{R}^2$) such that the triangle inequality achieves **equality**, i.e. $\lVert \vec u + \vec v \rVert = \lVert \vec u \rVert + \lVert \vec v \rVert$.

:::

## Draft Solution

Equality in
\[
\|\vec u+\vec v\|\le \|\vec u\|+\|\vec v\|
\]
happens when vectors point in the same direction.

Example:
\[
\vec u=\begin{bmatrix}1\\2\end{bmatrix},\quad
\vec v=\begin{bmatrix}2\\4\end{bmatrix}=2\vec u.
\]
Then
\[
\|\vec u+\vec v\|=\|3\vec u\|=3\|\vec u\|=\|\vec u\|+\|\vec v\|.
\]
