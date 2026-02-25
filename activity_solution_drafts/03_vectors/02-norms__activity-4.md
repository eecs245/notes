---
notebook: 03_vectors/02-norms.ipynb
activity: 4
cell_index: 15
---

# Activity 4

This is Activity 4.

Source notebook: [`03_vectors/02-norms.ipynb`](../../03_vectors/02-norms.ipynb)

## Activity Text

:::{tip} Activity 4
:class: dropdown

In the case of $\vec x = \begin{bmatrix} 12 \\ 5 \end{bmatrix}$, the $L_1$ norm ($12 + 5 = 17$) is greater than the $L_2$ norm ($\sqrt{12^2 + 5^2} = \sqrt{169} = 13$), i.e. $\lVert \vec x \rVert_1 > \lVert \vec x \rVert_2$.

1. Find a vector $\vec y \in \mathbb{R}^2$ such that $\lVert \vec y \rVert_1 = \lVert \vec y \rVert_2$.
1. Try and find a vector $\vec z \in \mathbb{R}^2$ such that $\lVert \vec z \rVert_1 < \lVert \vec z \rVert_2$. What do you encounter?
1. **Prove** that $\lVert \vec x \rVert_2 \leq \sqrt{n} \lVert \vec x \rVert_\infty$ for any vector $\vec x \in \mathbb{R}^n$. (Hint: Start with the definition of the $L_2$ norm of $\vec x$, square it, and try and compare each element in the sum to the largest element in the vector.)

:::

## Draft Solution

1. Example with equality \(\|\vec y\|_1=\|\vec y\|_2\):
\[
\vec y=\begin{bmatrix}1\\0\end{bmatrix}
\Rightarrow
\|\vec y\|_1=1,\ \|\vec y\|_2=1.
\]
(Any vector with exactly one non-zero component works.)

2. Trying to find \(\vec z\) with \(\|\vec z\|_1<\|\vec z\|_2\) fails: this cannot happen. In fact,
\[
\|\vec z\|_2\le \|\vec z\|_1
\]
for all vectors.

3. Proof of \(\|\vec x\|_2\le \sqrt n\|\vec x\|_\infty\):
\[
\|\vec x\|_2^2=\sum_{i=1}^n x_i^2
\le \sum_{i=1}^n \|\vec x\|_\infty^2
= n\|\vec x\|_\infty^2.
\]
Take square roots:
\[
\boxed{\|\vec x\|_2\le \sqrt n\|\vec x\|_\infty}.
\]
