---
notebook: 04_linear_independence/04-lines-planes-hyperplanes.ipynb
activity: 5
cell_index: 20
---

# Activity 5

This is Activity 5.

Source notebook: [`04_linear_independence/04-lines-planes-hyperplanes.ipynb`](../../04_linear_independence/04-lines-planes-hyperplanes.ipynb)

## Activity Text

:::{tip} Activity 5
:class: dropdown
Verify that the cross product of $\color{orange} \vec u$ and $\color{#3d81f6} \vec v$ is orthogonal to both $\color{orange} \vec u$ and $\color{#3d81f6} \vec v$.
:::

## Draft Solution

Using the vectors from the notebook,
\[
\vec u=\begin{bmatrix}5\\2\\1\end{bmatrix},
\quad
\vec v=\begin{bmatrix}-2\\3\\0\end{bmatrix},
\quad
\vec u\times\vec v=\begin{bmatrix}-3\\-2\\19\end{bmatrix}.
\]
Check orthogonality by dot products:
\[
(\vec u\times\vec v)\cdot\vec u
=(-3)(5)+(-2)(2)+19(1)=-15-4+19=0,
\]
\[
(\vec u\times\vec v)\cdot\vec v
=(-3)(-2)+(-2)(3)+19(0)=6-6+0=0.
\]
So \(\vec u\times\vec v\) is orthogonal to both \(\vec u\) and \(\vec v\).
