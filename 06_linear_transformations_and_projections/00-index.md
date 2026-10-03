---
numbering: false
---

# Chapter 6: Linear Transformations and Projections

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 6.1 Linear Transformations

- $T: \mathbb{R}^d \to \mathbb{R}^n$ is **linear** if $T(\vec x + \vec y) = T(\vec x) + T(\vec y)$ and $T(c\vec x) = cT(\vec x)$. Every linear $T$ is $T(\vec x) = A\vec x$, where column $j$ of $A$ is $T(\vec e_j)$. Linear maps send $\vec 0$ to $\vec 0$; $A\vec x + \vec b$ is **affine**, not linear if $\vec b \ne \vec 0$.
- A $2 \times 2$ matrix maps the unit square to the parallelogram spanned by its columns. $AB\vec x$ applies $B$ first, then $A$. Counterclockwise rotation by $\theta$: $R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$.
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc \qquad\quad \det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = a(ei - fh) - b(di - fg) + c(dh - eg)$$

- $\det(A)$ is the signed area (volume) that the unit square (cube) maps to, and $\det(A) = 0 \iff$ the columns are dependent. $\det(A^T) = \det(A)$, $\det(AB) = \det(A)\det(B)$, $\det(cA) = c^n\det(A)$, a column swap flips the sign, and orthogonal $Q$ has $\det(Q) = \pm 1$.

### 6.2 Inverses

- **Invertible matrix theorem** (square $A$; $AA^{-1} = A^{-1}A = I$): invertible $\iff$ rank $n$ $\iff$ independent columns (rows) $\iff$ $\text{nullsp}(A) = \{\vec 0\}$ $\iff$ $\det(A) \ne 0$ $\iff$ $A\vec x = \vec b$ always has exactly one solution.

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix} \qquad (AB)^{-1} = B^{-1}A^{-1} \qquad (A^T)^{-1} = (A^{-1})^T \qquad Q^{-1} = Q^T$$

- $X^TX$ is invertible $\iff$ $X$'s columns are independent.

### 6.3 Projecting onto the Column Space

- To minimize $\lVert \vec y - X\vec w \rVert^2$, make $\vec e = \vec y - X\vec w^*$ **orthogonal to $\text{colsp}(X)$** ($X^T\vec e = \vec 0$):

$$X^TX\vec w^* = X^T\vec y \;\; \text{(normal equations)} \qquad\quad \vec w^* = (X^TX)^{-1}X^T\vec y \;\; \text{if } \text{rank}(X) = d$$

- $\vec p = X\vec w^*$ is the projection of $\vec y$ onto $\text{colsp}(X)$; $\vec e \in \text{nullsp}(X^T)$. If $\vec 1 \in \text{colsp}(X)$, the errors sum to 0.

### 6.4 The Complete Solution to the Normal Equations

- The normal equations always have a solution. With dependent columns there are infinitely many, $\vec w_s + \vec n$ for $\vec n \in \text{nullsp}(X)$, all with the **same** $\vec p$; get $\vec w_s$ by solving without the dependent columns, then putting 0 in their slots.
- **Projection matrix:** $P = X(X^TX)^{-1}X^T$ ($n \times n$), $\vec p = P\vec y$. $P^T = P$, $P^2 = P$, $PX = X$, $\text{rank}(P) = \text{rank}(X)$; not invertible unless $P = I$.
- Rotation: orthogonal, $\det = 1$. Reflection ($I - 2\vec u\vec u^T$, unit $\vec u$): orthogonal, $\det = -1$. Projection: symmetric and idempotent.

### 6.5 The Gram-Schmidt Process

$$\vec Q_k = \vec v_k - \sum_{j < k} \frac{\vec v_k \cdot \vec Q_j}{\vec Q_j \cdot \vec Q_j}\,\vec Q_j \qquad\quad \vec q_k = \frac{\vec Q_k}{\lVert \vec Q_k \rVert}$$

- Turns independent $\vec v$'s into orthonormal $\vec q$'s with the same span (project onto the **new** $\vec Q_j$'s). With the $\vec q$'s as the columns of $Q$: $Q^TQ = I$, the weights on $Q$'s columns are $Q^T\vec y$, and $\vec p = QQ^T\vec y$.
