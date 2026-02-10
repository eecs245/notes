---
notebook: 03_vectors/02-norms.ipynb
activity: 5
cell_index: 16
---

# Activity 5

This is Activity 5.

Source notebook: [`03_vectors/02-norms.ipynb`](../../03_vectors/02-norms.ipynb)

## Activity Text

:::{tip} Activity 5
:class: dropdown

In the cell above:

1. Write `u ** 2`. Squaring a vector is not an operation we've discussed (and isn't an operation that exists in math), but `numpy` gives you back another array. What does this array contain?
1. Using `np.sum` and the new array you just created, find the norm of `u` without using `np.linalg.norm`.
1. Find the norm of `3 * u - 0.5 * v` using the same technique, and make sure you get the same result as was already displayed for you.
:::

## Draft Solution

From the embedded cell:
\[
u=\begin{bmatrix}3\\1\end{bmatrix},\quad v=\begin{bmatrix}4\\-6\end{bmatrix}.
\]

1. `u ** 2` gives element-wise squares:
\[
[3^2,1^2]=[9,1].
\]

2. Norm of \(u\) via `np.sum`:
\[
\|u\|=\sqrt{\sum (u**2)}=\sqrt{9+1}=\sqrt{10}.
\]

3. Compute
\[
3u-0.5v=\begin{bmatrix}9\\3\end{bmatrix}-\begin{bmatrix}2\\-3\end{bmatrix}
=\begin{bmatrix}7\\6\end{bmatrix}.
\]
Then
\[
\|3u-0.5v\|=\sqrt{7^2+6^2}=\sqrt{85}.
\]
