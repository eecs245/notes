---
notebook: 03_vectors/04-projecting-onto-a-single-vector.ipynb
activity: 1
cell_index: 15
---

# Activity 1

This is Activity 1.

Source notebook: [`03_vectors/04-projecting-onto-a-single-vector.ipynb`](../../03_vectors/04-projecting-onto-a-single-vector.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown

Let $\vec v_1 = \begin{bmatrix} -1 \\ 2 \\ 2 \end{bmatrix}$, $\vec v_2 = \begin{bmatrix} 2 \\ 2 \\ -1 \end{bmatrix}$, and $\vec v_3 = \begin{bmatrix} 2 \\ -1 \\ 2 \end{bmatrix}$.

Write $\vec u = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$ as a linear combination of $\vec v_1$, $\vec v_2$, and $\vec v_3$, **without** solving a system of equations.

:::

## Draft Solution

Notice the three vectors are pairwise orthogonal and each has norm squared 9:
\[
\vec v_i\cdot\vec v_j=0\ (i\ne j),\quad \vec v_i\cdot\vec v_i=9.
\]
So for orthogonal decomposition,
\[
\vec u=\sum_{i=1}^3 \frac{\vec u\cdot\vec v_i}{\vec v_i\cdot\vec v_i}\,\vec v_i.
\]
Now
\[
\vec u\cdot\vec v_1=\vec u\cdot\vec v_2=\vec u\cdot\vec v_3=3,
\]
so each coefficient is \(3/9=1/3\). Therefore
\[
\boxed{\vec u=\frac13\vec v_1+\frac13\vec v_2+\frac13\vec v_3}.
\]
