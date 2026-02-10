---
notebook: 05_matrices/01-matrix-operations.ipynb
activity: 1
cell_index: 3
---

# Activity 1

This is Activity 1.

Source notebook: [`05_matrices/01-matrix-operations.ipynb`](../../05_matrices/01-matrix-operations.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown

In the cell above, try and find the bottom-right entry of `A`, first using positive indexing and then using negative indexing.
:::

## Draft Solution

From the code cell,
\[
A=\begin{bmatrix}3&1&4\\2&1&9\\0&-1&0\\2&-2&0\end{bmatrix}.
\]
Bottom-right entry is row 4, column 3 (1-indexed), i.e. value 0.

- Positive indexing in NumPy: `A[3, 2]`.
- Negative indexing: `A[-1, -1]`.

Both give \(\boxed{0}\).
