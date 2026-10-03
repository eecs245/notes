---
numbering: false
---

# Chapter 9: Eigenvalues and Eigenvectors

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 9.1 Eigenvalues and Eigenvectors

- For square $A$, a **nonzero** $\vec v$ is an **eigenvector** with **eigenvalue** $\lambda$ if $A\vec v = \lambda\vec v$: $A$ keeps $\vec v$ on its line through the origin, scaled by $\lambda$ (flipped if $\lambda < 0$). Every $c\vec v$ with $c \ne 0$ is an eigenvector too.
- $A^k\vec v = \lambda^k\vec v$. $0$ is an eigenvalue iff $A$ isn't invertible.

### 9.2 The Characteristic Polynomial

$$\lambda \text{ is an eigenvalue} \iff (A - \lambda I)\vec v = \vec 0 \text{ for some } \vec v \ne \vec 0 \iff \det(A - \lambda I) = 0$$

- The eigenvalues are the $n$ roots of $p(\lambda) = \det(A - \lambda I)$, counting repeats, possibly complex (e.g. rotations). Eigenvectors for $\lambda$: the nonzero vectors in $\text{nullsp}(A - \lambda I)$, its **eigenspace**.
- $\sum \lambda_i = \text{trace}(A)$ (the diagonal sum) and $\prod \lambda_i = \det(A)$; for $2 \times 2$, $p(\lambda) = \lambda^2 - \text{trace}(A)\lambda + \det(A)$.
- A triangular matrix's eigenvalues are its diagonal entries. $A$ and $A^T$ have the same eigenvalues.

### 9.3 Markov Chains and Adjacency Matrices

- Adjacency matrix: $a_{ij} = P(\text{state } j \to \text{state } i)$, so the columns sum to 1, and $\vec x_k = A^k\vec x_0$.
- **Steady state** (long-run distribution): an eigenvector for $\lambda = 1$, scaled to sum to 1. Eigenvalue 1 always exists, since $A^T\vec 1 = \vec 1$, and every $|\lambda| \le 1$.
- If $\vec x = \sum c_i\vec v_i$ (independent eigenvectors), $A^k\vec x = \sum c_i\lambda_i^k\vec v_i$: the **dominant** (largest $|\lambda|$) term wins, so the direction of $A^k\vec x$ approaches its eigenvector (**power method**).

### 9.4 Multiplicities and Diagonalization

$$A = V\Lambda V^{-1}, \quad V = \begin{bmatrix} \vec v_1 & \cdots & \vec v_n \end{bmatrix}, \quad \Lambda = \text{diag}(\lambda_1, \dots, \lambda_n) \text{ in the same order} \qquad A^k = V\Lambda^kV^{-1}$$

- This needs $n$ independent eigenvectors (**diagonalizable**). $V^{-1}$: to the eigenbasis; $\Lambda$: scale; $V$: back.
- **AM**($\lambda$) is its exponent in $p(\lambda)$; **GM**($\lambda$) $= \dim(\text{nullsp}(A - \lambda I)) = n - \text{rank}(A - \lambda I)$. $1 \le \text{GM} \le \text{AM}$; diagonalizable iff GM = AM for all $\lambda$.
- Eigenvectors for distinct eigenvalues are independent, so $n$ distinct real eigenvalues $\Rightarrow$ diagonalizable. Diagonalizable and invertible are unrelated; symmetric $\Rightarrow$ diagonalizable.

### 9.5 Symmetric Matrices and the Spectral Theorem

- **Spectral theorem:** a symmetric $A$ has real eigenvalues, orthogonal eigenvectors for different eigenvalues, and $A = Q\Lambda Q^T$ with $Q$ orthogonal: orthonormal eigenvectors as columns (for a repeated $\lambda$, an orthonormal basis of its eigenspace).
- $Q\Lambda Q^T\vec x$: rotate, scale, rotate back; unit circle $\to$ ellipse, semi-axes $|\lambda_i|$ along eigenvectors.

### 9.6 Positive Semidefinite Matrices and the Rayleigh Quotient

- For symmetric $A$, $\vec x^TA\vec x = \sum_i \lambda_i y_i^2$ ($\vec y = Q^T\vec x$): **PSD** ($\vec x^TA\vec x \ge 0$ for all $\vec x$) iff all $\lambda_i \ge 0$; **PD** ($> 0$ for $\vec x \ne \vec 0$) iff all $\lambda_i > 0$.
- $f$ is convex iff its Hessian is PSD everywhere; $\vec x^TA\vec x$ has Hessian $2A$, so it's convex iff $A$ is PSD. $X^TX$ is PSD, so $X^TX + \lambda I$ ($\lambda > 0$) has eigenvalues $\ge \lambda$: invertible.

$$g(\vec v) = \vec v^TA\vec v \,/\, \vec v^T\vec v \;\; \text{(Rayleigh quotient)} \qquad\quad \lambda_{\min} \le g(\vec v) \le \lambda_{\max}$$

- $g$ depends only on direction ($= \vec v^TA\vec v$ if $\lVert \vec v \rVert = 1$). Its critical points are eigenvectors; the max (min) is at the eigenvector for $\lambda_{\max}$ ($\lambda_{\min}$). Split cross terms evenly: $12xy \to$ 6 in both off-diagonal entries.
