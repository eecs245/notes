---
notebook: 04_linear_independence/03-vector-spaces-basis-dimension.ipynb
activity: 3
cell_index: 17
---

# Activity 3

This is Activity 3.

Source notebook: [`04_linear_independence/03-vector-spaces-basis-dimension.ipynb`](../../04_linear_independence/03-vector-spaces-basis-dimension.ipynb)

## Activity Text

:::{tip} Activity 3
:class: dropdown

For each of the following subspaces, find **two** possible bases and state the dimension.

1. The subspace of $\mathbb{R}^3$ containing vectors $\begin{bmatrix} x \\ y \\ z \end{bmatrix}$ such that $2x - 3y + 4z = 0$.
1. The subspace of $\mathbb{R}^6$ containing vectors in which the third and fourth coordinates are 0.
:::

## Draft Solution

1. Subspace in \(\mathbb R^3\): \(2x-3y+4z=0\).

Dimension is 2 (one linear constraint in \(\mathbb R^3\)).

Two possible bases:
- Basis A: \(\{[3,2,0]^T,\ [-2,0,1]^T\}\).
- Basis B: \(\{[3,2,0]^T,\ [4,0,-2]^T\}\).

2. Subspace in \(\mathbb R^6\) with 3rd and 4th coordinates zero:
\[
\{[a,b,0,0,c,d]^T: a,b,c,d\in\mathbb R\}.
\]
Dimension is 4.

Two possible bases:
- Basis A: \(\{e_1,e_2,e_5,e_6\}\).
- Basis B: \(\{e_1+e_2,\ e_2,\ e_5+e_6,\ e_6\}\).
