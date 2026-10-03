---
numbering: false
---

# Chapter 4: Linear Independence

## Sections

:::{toc}
:context: children
:depth: 1
:::

## Summary

### 4.1 Span

- The **span** of a set of vectors is the set of all of their linear combinations: $$\text{span}(\{\vec v_1, \dots, \vec v_d\}) = \{a_1\vec v_1 + \cdots + a_d\vec v_d \mid a_1, \dots, a_d \in \mathbb{R}\}$$
- One nonzero vector spans a line through $\vec 0$.
- Two non-collinear vectors (i.e. vectors that point in different directions) span a plane through $\vec 0$.
- $d$ vectors in $\mathbb{R}^n$ span a subspace of dimension between $0$ and $\min(d, n)$.

### 4.2 Linear Independence

- $\vec v_1, \dots, \vec v_d$ are **linearly independent** if none is a linear combination of the others; equivalently, $a_1\vec v_1 + \cdots + a_d\vec v_d = \vec 0$ only when every $a_i = 0$.
- If they are **linearly dependent**, at least one (not necessarily every) vector is a combination of the others.
- If $\vec v_1, \dots, \vec v_d$ are linearly independent, every $\vec b$ in their span can only be written in one way as a linear combination of the $\vec v_i$'s; otherwise, every $\vec b$ can be written in infinitely many ways.
- Nonzero, pairwise orthogonal vectors are independent; a set containing $\vec 0$ is dependent.
- Independence needs $d \le n$, spanning $\mathbb{R}^n$ needs $d \ge n$. For exactly $n$ vectors in $\mathbb{R}^n$, the following are equivalent:
    - they are linearly independent;
    - they span $\mathbb{R}^n$;
    - they form a basis for $\mathbb{R}^n$.
- **Algorithm to find a basis:** in order, keep each nonzero $\vec v_i$ that isn't a combination of the kept ones. The result is a **basis** for the span (its size is the dimension); which vectors get kept can depend on order.

### 4.3 Vector Spaces, Basis, and Dimension

- **Vector space:** collection of objects (vectors) that areclosed under addition and scalar multiplication (plus 8 properties), e.g. $\mathbb{R}^n$.
- A **subspace** $S$ of $V$ satisfies:
    - $\vec 0 \in S$;
    - $\vec u, \vec v \in S \Rightarrow \vec u + \vec v \in S$;
    - $\vec u \in S, c \in \mathbb{R} \Rightarrow c\vec u \in S$.<br>Think of a subspace as a closed neighborhood within a vector space.
- Every span is a subspace, and every subspace is a span.
- A **basis** for $S$ spans $S$ and is independent (remove a vector: it stops spanning; add one: it's dependent). Bases aren't unique, but all have the same size, $\dim(S)$.
- The **standard basis** $\vec e_1, \dots, \vec e_n$ tells us $\dim(\mathbb{R}^n) = n$.

### 4.4 Lines, Planes, Hyperplanes, and the Cross Product

- The **parametric form** of a line is $$\vec p_0 + t\vec v$$
- The parametric form of a plane is $$\vec p_0 + s\vec u + t\vec v$$ with $\vec u, \vec v$ independent.
- A line/plane is a subspace only if it passes through $\vec 0$. A shifted subspace is called an **affine** subspace.
- A plane in $\mathbb{R}^3$ can be described by the equation $$ax + by + cz + d = 0$$ where $\begin{bmatrix}a \\ b \\ c \end{bmatrix}$ is a normal vector to the plane.
- One way to find a normal vector to a plane is to take the cross product of two linearly independent vectors $\vec u, \vec v$ that lie on the plane:
$$\vec u \times \vec v = \begin{bmatrix} u_2v_3 - u_3v_2 \\ u_3v_1 - u_1v_3 \\ u_1v_2 - u_2v_1 \end{bmatrix} \qquad \text{(only in } \mathbb{R}^3\text{; orthogonal to both } \vec u \text{ and } \vec v\text{)}$$
- A **hyperplane** in $\mathbb{R}^n$ is $\vec a \cdot \vec x + b = 0$ (normal $\vec a \ne \vec 0$), an $(n-1)$-dimensional slice: a line in $\mathbb{R}^2$, a plane in $\mathbb{R}^3$. It splits $\mathbb{R}^n$ into $\vec a \cdot \vec x + b > 0$ and $< 0$, like a linear classifier's boundary.
