---
notebook: 03_vectors/01-vectors-and-linear-combinations.ipynb
activity: 1
cell_index: 7
---

# Activity 1

This is Activity 1.

Source notebook: [`03_vectors/01-vectors-and-linear-combinations.ipynb`](../../03_vectors/01-vectors-and-linear-combinations.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown

As we'll see later in this section, a **unit vector** is a vector with norm 1. It's common to use unit vectors to describe directions. For instance, there are infinitely many vectors in $\mathbb{R}^2$ that point in the same direction as $\vec v = \begin{bmatrix} 4 \\ -6 \end{bmatrix}$ from above, like $\begin{bmatrix} 2 \\ -3 \end{bmatrix}$ and $\begin{bmatrix} 40 \\ -60 \end{bmatrix}$. (If you don't believe me, draw it out!)

Find a unit vector that points in the same direction as the vector $\vec x = \begin{bmatrix} 12 \\ 5 \end{bmatrix}$, and verify that it has norm 1. Technically, to answer this, you'll need to use the fact that vectors can be multiplied by a scalar, which we haven't yet discussed, but see how far your intuition takes you!

:::

## Draft Solution

For \(\vec x=\begin{bmatrix}12\\5\end{bmatrix}\),
\[
\|\vec x\|=\sqrt{12^2+5^2}=\sqrt{169}=13.
\]
So a unit vector in the same direction is
\[
\hat x=\frac{1}{13}\begin{bmatrix}12\\5\end{bmatrix}
=\begin{bmatrix}12/13\\5/13\end{bmatrix}.
\]
Check:
\[
\|\hat x\|=\sqrt{(12/13)^2+(5/13)^2}=\sqrt{169/169}=1.
\]
