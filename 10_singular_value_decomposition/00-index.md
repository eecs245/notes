---
numbering: false
---

# Chapter 10: Singular Value Decomposition

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 10.1 Computing the Singular Value Decomposition

$$X = U\Sigma V^T \qquad X\vec v_i = \sigma_i\vec u_i \qquad X^T\vec u_i = \sigma_i\vec v_i \qquad X^TX = V\Sigma^T\Sigma V^T \qquad XX^T = U\Sigma\Sigma^TU^T$$

| Piece | Shape | Contents |
|---|---|---|
| $U$ | $n \times n$, orthogonal | left singular vectors: eigenvectors of $XX^T$ |
| $\Sigma$ | $n \times d$ (same as $X$) | $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$ on the diagonal, then zeros |
| $V$ | $d \times d$, orthogonal | right singular vectors: eigenvectors of $X^TX$ |

- Every $X$ has an SVD; $\text{rank}(X)$ of the $\sigma_i$ are nonzero; $X^TX$, $XX^T$ share nonzero eigenvalues $\sigma_i^2$.
- **By hand:** $\sigma_i = \sqrt{\lambda_i(X^TX)}$, decreasing; $X^TX$'s orthonormal eigenvectors (same order) form $V$; $\vec u_i = \frac{1}{\sigma_i}X\vec v_i$ for $\sigma_i > 0$; an orthonormal basis of $\text{nullsp}(X^T)$ completes $U$ (wide $X$: start from $XX^T$). So $U \ne V$ in general, and a pair $\vec u_i, \vec v_i$ may flip sign together.
- Symmetric $X$: the $\sigma$'s are the $|\lambda|$'s, re-sorted; PSD $X$: $Q\Lambda Q^T$ is an SVD; orthogonal $X$: all $\sigma_i = 1$; otherwise no simple link to the $\lambda$'s. $X\vec w = U(\Sigma(V^T\vec w))$: rotate or reflect, stretch (padding or dropping coordinates), rotate or reflect.

### 10.2 Low-Rank Approximation

$$X = U_r\Sigma_rV_r^T = \sum_{i=1}^r \sigma_i\vec u_i\vec v_i^T \qquad\quad X_k = \sum_{i=1}^k \sigma_i\vec u_i\vec v_i^T \;\; \text{(rank-}k\text{ approximation)}$$

- Compact SVD: $U_r$ ($n \times r$) and $V_r$ ($d \times r$) with $U_r^TU_r = V_r^TV_r = I_r$ (orthogonal only if square). Bases: $\vec u_1, \dots, \vec u_r$ for $\text{colsp}(X)$, $\vec v_1, \dots, \vec v_r$ for the row space; the rest of $U$ and $V$ for $\text{nullsp}(X^T)$ and $\text{nullsp}(X)$.
- Each $\sigma_i\vec u_i\vec v_i^T$ is a rank-one outer product; storing $X_k$ takes $k(1 + n + d)$ numbers, not $nd$.

### 10.3 The Best Direction

- **Goal:** uncorrelated new features (linear combinations of the old) that keep the most information. First **center**: $\tilde X$ is $X$ minus its column means, so every $\tilde X\vec w$ has mean 0.
- The best unit $\vec v$ minimizes the mean squared **orthogonal** error (regression uses vertical error), i.e. maximizes the projected variance:

$$\frac{1}{n}\sum_{i=1}^n \lVert \tilde x_i - (\tilde x_i \cdot \vec v)\vec v \rVert^2 = \underbrace{\frac{1}{n}\sum_{i=1}^n \lVert \tilde x_i \rVert^2}_{\text{total variance}} - \underbrace{\frac{1}{n}\lVert \tilde X\vec v \rVert^2}_{\text{PV}(\vec v)}$$

- Maximize the Rayleigh quotient $\lVert \tilde X\vec v \rVert^2 / \lVert \vec v \rVert^2$: critical points are eigenvectors of $\tilde X^T\tilde X$; max $\lambda_1 = \sigma_1^2$ at $\vec v_1$.

### 10.4 Principal Components Analysis

$$\text{PC}_j = \tilde X\vec v_j = \sigma_j\vec u_j \qquad \tilde XV = U\Sigma \qquad \text{Var}(\text{PC}_j) = \frac{\sigma_j^2}{n} \qquad \text{total variance} = \sum_j \frac{\sigma_j^2}{n}$$

- **Recipe:** center, then take the SVD $\tilde X = U\Sigma V^T$; $V$'s columns are the directions, best first. The PCs are uncorrelated: mean 0, and $\text{PC}_i \cdot \text{PC}_j = \sigma_i\sigma_j\,\vec u_i \cdot \vec u_j = 0$ ($i \ne j$).
- Proportion explained by PC $j$: $\sigma_j^2 / \sum_{i=1}^r \sigma_i^2$ (sum over $j \le k$ for $k$ PCs; a scree plot helps pick $k$).
- PCA is unsupervised and **scale-sensitive**: a big-spread feature (e.g. in grams) dominates PC 1.
