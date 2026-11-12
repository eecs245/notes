---
numbering: false
---

# Chapter 5: Matrices

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 5.1 Matrix Operations

- $A \in \mathbb{R}^{n \times d}$ has $n$ rows and $d$ columns; $A_{ij}$ is in row $i$, column $j$. Adding and scaling are entrywise.
- **Golden rule:** $AB$ is defined only when the inner dimensions match: $(n \times d)(d \times p) = n \times p$.

- **Interpretation 1:** $A\vec x$ is a linear combination of $A$'s columns, $\vec a^{(1)}, \vec a^{(2)}, \dots, \vec a^{(d)}$, weighted by the entries in $\vec x$:

  $$A\vec x = x_1\vec a^{(1)} + \cdots + x_d\vec a^{(d)}$$

- **Interpretation 2:** $A\vec x$ contains the dot products of $A$'s rows with $\vec x$:

  $$(A\vec x)_i = (\text{row } i \text{ of } A) \cdot \vec x$$

- Matrix multiplication is **associative**, $(AB)C = A(BC)$, and **distributive**, $A(B + C) = AB + AC$, but generally **not commutative**, $AB \ne BA$.
- In NumPy, use `A @ B` for matrix multiplication, not `A * B`.

### 5.2 Transpose and Special Matrices

- $(A^T)_{ij} = A_{ji}$; $(A + B)^T = A^T + B^T$; $(AB)^T = B^TA^T$ (order reverses).
- The **dot product** $\vec u \cdot \vec v = \vec u^T\vec v$ is a scalar, but $\vec u\vec v^T$ is a matrix called the **outer product** of $\vec u$ and $\vec v$.
- $\lVert A\vec x \rVert^2 = \vec x^TA^TA\vec x$.
- **Identity:** $I\vec x = \vec x$, $IA = AI = A$. **Diagonal:** $A_{ij} = 0$ for $i \ne j$ (scales each component). **Upper (lower) triangular:** square, zeros below (above) the diagonal.
- **Symmetric:** $A = A^T$. $X^TX$ is always symmetric ($d \times d$), with entry $(i, j)$ equal to $\vec x^{(i)} \cdot \vec x^{(j)}$, where $\vec x^{(i)}$ is the $i$-th column of $X$.
- **Orthogonal:** square with $A^TA = AA^T = I$ (orthonormal columns and rows); preserves length, $\lVert A\vec x \rVert = \lVert \vec x \rVert$. A non-square $A$ with $A^TA = I$ only has orthonormal columns.

### 5.3 Rank and Column Space

$$\text{colsp}(A) = \underbrace{\{A\vec x \mid \vec x \in \mathbb{R}^d\}}_{\text{subspace of }\mathbb{R}^n}$$

$$\text{rowsp}(A) = \text{colsp}(A^T) = \underbrace{\{A^T\vec y \mid \vec y \in \mathbb{R}^n\}}_{\text{subspace of }\mathbb{R}^d}$$

- $\text{rank}(A)$ = maximum number of linearly independent columns = $\dim(\text{colsp}(A))$ = maximum number of linearly independent rows, so $0 \le \text{rank}(A) \le \min(n, d)$.
- **Full rank:** $\text{rank}(A) = \min(n, d)$.
- One way to create a basis for $\text{colsp}(A)$: keep each column that isn't a combination of the columns kept before it. (Remember, every subspace has infinitely many bases.)
- A $2 \times 2$ matrix with rows $(a, b)$ and $(c, d)$ has rank 2 iff $ad - bc \ne 0$. A diagonal matrix's rank is its number of nonzero diagonal entries.
- $\vec u\vec v^T$ has rank 1 for nonzero $\vec u, \vec v$.

### 5.4 Null Space and the Rank-Nullity Theorem

- The **null space** is the set of inputs that get sent to $\vec 0$ by $A$:

  $$\text{nullsp}(A) = \{\vec x \in \mathbb{R}^d \mid A\vec x = \vec 0\}$$

- **Rank-nullity theorem:**

  $$\text{rank}(A) + \dim(\text{nullsp}(A)) = d \;\; (\text{number of columns})$$

- Linearly independent columns $\iff \text{nullsp}(A) = \{\vec 0\}$.
- Column relationships give null space vectors, e.g. if col 3 = col 1 − col 2, then

  $$A\begin{bmatrix} 1 \\ -1 \\ -1 \end{bmatrix} = \vec 0$$

| Subspace (assuming $\text{rank}(A) = r$) | Lives in | Dimension | Orthogonal complement |
|---|---|---|---|
| column space $\text{colsp}(A)$ | $\mathbb{R}^n$ | $r$ | left null space |
| row space $\text{colsp}(A^T)$ | $\mathbb{R}^d$ | $r$ | null space |
| null space $\text{nullsp}(A)$ | $\mathbb{R}^d$ | $d - r$ | row space |
| left null space $\text{nullsp}(A^T)$ | $\mathbb{R}^n$ | $n - r$ | column space |

- $\text{rank}(AB) \le \min(\text{rank}(A), \text{rank}(B))$.
- $\text{rank}(X^TX) = \text{rank}(X)$: they share a null space, since $X^TX\vec v = \vec 0 \Rightarrow \lVert X\vec v \rVert^2 = \vec v^TX^TX\vec v = 0$.
- **CR decomposition** $A = CR$: $C$ ($n \times r$) is $A$'s linearly independent columns; column $j$ of $R$ ($r \times d$) builds column $j$ of $A$ from them. It shows row rank = column rank.
