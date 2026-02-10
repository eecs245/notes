---
notebook: 03_vectors/03-dot-product.ipynb
activity: 7
cell_index: 14
---

# Activity 7

This is Activity 7.

Source notebook: [`03_vectors/03-dot-product.ipynb`](../../03_vectors/03-dot-product.ipynb)

## Activity Text

:::{tip} Activity 7
:class: dropdown

Let $\vec u = \begin{bmatrix} \sqrt{a} \\ \sqrt{b} \end{bmatrix}$ and $\vec v = \begin{bmatrix} \sqrt{b} \\ \sqrt{a} \end{bmatrix}$.

Using the Cauchy-Schwarz inequality, prove that the geometric mean of $a$ and $b$ is less than or equal to the arithmetic mean of $a$ and $b$.

:::

## Draft Solution

Assume \(a,b\ge 0\) so square roots are real.

Using
\[
\vec u=\begin{bmatrix}\sqrt a\\\sqrt b\end{bmatrix},
\quad
\vec v=\begin{bmatrix}\sqrt b\\\sqrt a\end{bmatrix},
\]
Cauchy-Schwarz gives
\[
|\vec u\cdot\vec v|\le \|\vec u\|\,\|\vec v\|.
\]
Compute each side:
\[
\vec u\cdot\vec v = \sqrt a\sqrt b+\sqrt b\sqrt a=2\sqrt{ab},
\]
\[
\|\vec u\|=\|\vec v\|=\sqrt{a+b}.
\]
So
\[
2\sqrt{ab}\le a+b
\quad\Longrightarrow\quad
\sqrt{ab}\le \frac{a+b}{2}.
\]
That is AM-GM.
