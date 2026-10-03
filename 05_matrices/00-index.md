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
- **Shape rule:** $AB$ is defined only when the inner dimensions match: $(n \times d)(d \times p) = n \times p$.

$$A\vec x = x_1\vec a^{(1)} + \cdots + x_d\vec a^{(d)} \qquad (A\vec x)_i = (\text{row } i \text{ of } A) \cdot \vec x \qquad (AB)_{ij} = (\text{row } i \text{ of } A) \cdot (\text{column } j \text{ of } B)$$

- $A\vec x$ is a **linear combination of $A$'s columns** $\vec a^{(j)}$, weighted by $\vec x$. Column $j$ of $AB$ is $A$ (column $j$ of $B$).
- $(AB)C = A(BC)$ and $A(B + C) = AB + AC$, but $AB \ne BA$ in general. Use `A @ B`, not `A * B`.

### 5.2 Transpose and Special Matrices

- $(A^T)_{ij} = A_{ji}$; $(A + B)^T = A^T + B^T$; $(AB)^T = B^TA^T$ (order reverses). $\vec u \cdot \vec v = \vec u^T\vec v$ is a scalar, but $\vec u\vec v^T$ is a matrix. $\lVert A\vec x \rVert^2 = \vec x^TA^TA\vec x$.
- **Identity:** $I\vec x = \vec x$, $IA = AI = A$. **Diagonal:** $A_{ij} = 0$ for $i \ne j$ (scales each component). **Upper (lower) triangular:** square, zeros below (above) the diagonal.
- **Symmetric:** $A = A^T$. $X^TX$ is always symmetric ($d \times d$), with entry $(i, j)$ = (column $i$) $\cdot$ (column $j$).
- **Orthogonal:** square with $A^TA = AA^T = I$ (orthonormal columns and rows); preserves length, $\lVert A\vec x \rVert = \lVert \vec x \rVert$. A non-square $A$ with $A^TA = I$ only has orthonormal columns.

### 5.3 Rank and Column Space

$$\text{colsp}(A) = \{A\vec x \mid \vec x \in \mathbb{R}^d\} \subseteq \mathbb{R}^n \qquad\quad \text{rowsp}(A) = \text{colsp}(A^T) = \{A^T\vec y \mid \vec y \in \mathbb{R}^n\} \subseteq \mathbb{R}^d$$

- $\text{rank}(A)$ = # independent columns = $\dim(\text{colsp}(A))$ = # independent rows, so $0 \le \text{rank}(A) \le \min(n, d)$. **Full rank:** $\text{rank}(A) = \min(n, d)$.
- Basis for $\text{colsp}(A)$: keep each column that isn't a combination of the columns kept before it.
- A $2 \times 2$ matrix with rows $(a, b)$ and $(c, d)$ has rank 2 iff $ad - bc \ne 0$. A diagonal matrix's rank is its number of nonzero diagonal entries. $\vec u\vec v^T$ has rank 1 for nonzero $\vec u, \vec v$.

### 5.4 Null Space and the Rank-Nullity Theorem

$$\text{nullsp}(A) = \{\vec x \in \mathbb{R}^d \mid A\vec x = \vec 0\} \qquad\quad \text{rank}(A) + \dim(\text{nullsp}(A)) = d \;\; (\text{number of columns})$$

- Independent columns $\iff \text{nullsp}(A) = \{\vec 0\}$. Column relationships give null space vectors: col 3 = col 1 − col 2 means $A\begin{bmatrix} 1 & -1 & -1 \end{bmatrix}^T = \vec 0$.

| Subspace (rank $r$) | Lives in | Dimension | Orthogonal complement |
|---|---|---|---|
| column space $\text{colsp}(A)$ | $\mathbb{R}^n$ | $r$ | left null space |
| row space $\text{colsp}(A^T)$ | $\mathbb{R}^d$ | $r$ | null space |
| null space $\text{nullsp}(A)$ | $\mathbb{R}^d$ | $d - r$ | row space |
| left null space $\text{nullsp}(A^T)$ | $\mathbb{R}^n$ | $n - r$ | column space |

- $\text{rank}(AB) \le \min(\text{rank}(A), \text{rank}(B))$. $\text{rank}(X^TX) = \text{rank}(X)$: they share a null space, since $X^TX\vec v = \vec 0 \Rightarrow \lVert X\vec v \rVert^2 = \vec v^TX^TX\vec v = 0$.
- **CR decomposition** $A = CR$: $C$ ($n \times r$) is $A$'s independent columns; column $j$ of $R$ ($r \times d$) builds column $j$ of $A$ from them. It shows row rank = column rank and stores $r(n + d)$ numbers, not $nd$.
