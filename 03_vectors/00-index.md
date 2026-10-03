---
numbering: false
---

# Chapter 3: Vectors

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 3.1 Vectors and Linear Combinations

- A **vector** $\vec v \in \mathbb{R}^n$ is an ordered list of $n$ real numbers. $$\vec v = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}, \qquad \vec v \in \mathbb{R}^n$$
- A **linear combination** of $\vec v_1, \dots, \vec v_d$ is a vector of the form $$a_1\vec v_1 + a_2\vec v_2 + \cdots + a_d\vec v_d$$ for scalars $a_i$.

### 3.2 Norms
- The standard **norm** (also known as **length** or **magnitude**) is the $L_2$ norm:
$$\lVert \vec v \rVert = \sqrt{v_1^2 + \cdots + v_n^2}$$
- Other norms:
    - $L_1$ norm: $\lVert \vec v \rVert_1 = \sum_{i=1}^n |v_i|$.
    - $L_\infty$ norm: $\lVert \vec v \rVert_\infty = \max_i |v_i|$.
    - $L_p$ norm: $\lVert \vec v \rVert_p = \Big(\sum_{i=1}^n |v_i|^p\Big)^{1/p}$.
- A **unit vector** has norm 1; $$\frac{\vec v}{\lVert \vec v \rVert}$$ is the one in the direction of $\vec v \ne \vec 0$. "Norm" alone means $L_2$.

### 3.3 The Dot Product

- The **dot product** of two vectors $\vec u$ and $\vec v$ is a **scalar**. Two equivalent definitions:
$$\vec u \cdot \vec v = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$
$$\vec u \cdot \vec v = \lVert \vec u \rVert\,\lVert \vec v \rVert \cos\theta$$
- Note that $$\vec v \cdot \vec v = \lVert \vec v \rVert^2$$
- $\vec u$ and $\vec v$ are **orthogonal** if $$\vec u \cdot \vec v = 0$$
- The dot product is commutative and distributive, and $(c\vec u) \cdot \vec v = c(\vec u \cdot \vec v)$.
- **Expanding norms:** $$\lVert \vec u \pm \vec v \rVert^2 = \lVert \vec u \rVert^2 \pm 2\,\vec u \cdot \vec v + \lVert \vec v \rVert^2$$ For norm proofs: square, convert to dot products, and expand. If $\vec u \perp \vec v$: $\lVert \vec u + \vec v \rVert^2 = \lVert \vec u \rVert^2 + \lVert \vec v \rVert^2$.
- **Cosine similarity:** $$\cos\theta = \frac{\vec u \cdot \vec v}{\lVert \vec u \rVert \lVert \vec v \rVert}$$ ($\vec u, \vec v \ne \vec 0$), between $-1$ (opposite) and $1$ (same direction).
- **Cauchy-Schwarz**: $$\lvert \vec u \cdot \vec v \rvert \le \lVert \vec u \rVert\,\lVert \vec v \rVert$$ with equality if and only if one vector is a multiple of the other.

### 3.4 Projecting onto a Single Vector

- **Approximation problem:** which multiple $k\vec v$ ($\vec v \ne \vec 0$) is closest to $\vec u$, i.e. minimizes $\lVert \vec e \rVert = \lVert \vec u - k\vec v \rVert$? The one whose error is **orthogonal** to $\vec v$, giving the **orthogonal projection $\vec p$ of $\vec u$ onto $\vec v$**:

$$k^* = \frac{\vec u \cdot \vec v}{\vec v \cdot \vec v} \qquad\quad \vec p = \left(\frac{\vec u \cdot \vec v}{\vec v \cdot \vec v}\right)\vec v \qquad\quad \vec e = \vec u - \vec p, \quad \vec e \cdot \vec v = 0$$

```{figure} imgs/single-vector-projection.png
:class: notes-diagram
:alt: The original Chapter 3.4 plot: orange vector u, blue vector v, green projection p = k* v, and pink perpendicular error e = u - p, with rendered vector labels and a right-angle marker.
:width: 480px
```

- Projecting $\vec u$ onto $\vec v$ gives $\vec u = \vec p + \vec e$, where $\vec p$ and $\vec e$ are **orthogonal**.
- Onto a unit vector $\vec V$: $$\vec p = (\vec u \cdot \vec V)\,\vec V$$ Any nonzero multiple of $\vec v$ gives the same $\vec p$. $k^*$ can be negative.
- If $\vec v_1, \dots, \vec v_d$ are **mutually orthogonal** and $\vec u$ is a linear combination of them (i.e. is in their span), the coefficients are $$a_j = \frac{\vec u \cdot \vec v_j}{\vec v_j \cdot \vec v_j}$$ with no system to solve. Without orthogonality, this generally fails.
