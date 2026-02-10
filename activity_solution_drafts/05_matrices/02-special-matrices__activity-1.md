---
notebook: 05_matrices/02-special-matrices.ipynb
activity: 1
cell_index: 3
---

# Activity 1

This is Activity 1.

Source notebook: [`05_matrices/02-special-matrices.ipynb`](../../05_matrices/02-special-matrices.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown

**Activity 1.1**

In the cell above:

1. Define `x` to be an array corresponding to the vector $\vec x = \begin{bmatrix} 1 \\ 0 \\ 3 \end{bmatrix}$.
2. Find the norm of the product $A \vec x$ using `np.linalg.norm`.
3. Find the norm of the product $A \vec x$ using the fact that $\lVert A \vec x \rVert^2 = \vec x^T A^T A \vec x$, and verify that you get the same answer.

**Activity 1.2**

Suppose $M \in \mathbb{R}^{n \times d}$ is a matrix, $\vec{v} \in \mathbb{R}^d$ is a vector, and $s \in \mathbb{R}$ is a scalar.

Determine whether each of the following quantities is a matrix, vector, scalar, or undefined. If the result is a matrix or vector, determine its dimensions.

1. $M\vec{v}$

2. $\vec{v} M$

3. $\vec{v}^2$

4. $M^TM$

5. $MM^T$

6. $\vec{v}^T M \vec{v}$

7. $(sM\vec{v}) \cdot (sM\vec{v})$

8. $(s \vec{v}^T M^T)^T$

9. $\vec{v}^T M^T M \vec{v}$

10. $\vec{v}\vec{v}^T + M^TM$

11. $\frac{M \vec{v}}{\lVert \vec{v} \rVert} + (\vec{v}^T M^T M \vec{v}) M \vec{v}$

**Activity 1.3**

Let $A = \begin{bmatrix} 2 & 1 \\ 3 & 4 \\ -1 & 1 \end{bmatrix}$, $B = \begin{bmatrix} 1 & 0 & 2 \\ 2 & 1 & 3 \end{bmatrix}$, and $C = \begin{bmatrix} 1 & 0 & 2 & -1 \\ 0 & 1 & 1 & 1 \\ 1 & 1 & 0 & -1 \end{bmatrix}$.

1. Compute $AB$, then multiply the result by $C$.
1. Compute $A$, then multiply the result by $BC$. Do you get the same result as above? If so, what property of matrix multiplication guarantees this?
1. Determine a formula for $(ABC)^T$, and verify that your result works. (Hint: Start with the fact that $(AB)^T = B^T A^T$.)
:::

## Draft Solution

**Activity 1.1**

Using
\[
A=\begin{bmatrix}3&1&4\\2&1&9\\0&-1&0\\2&-2&0\end{bmatrix},
\quad
\vec x=\begin{bmatrix}1\\0\\3\end{bmatrix}.
\]

\[
A\vec x=
\begin{bmatrix}15\\29\\0\\2\end{bmatrix}.
\]
So
\[
\|A\vec x\|=\sqrt{15^2+29^2+0^2+2^2}=\sqrt{1070}.
\]

Also
\[
\|A\vec x\|^2=\vec x^T A^T A \vec x=1070,
\]
which gives the same norm \(\sqrt{1070}\).

**Activity 1.2** (\(M\in\mathbb R^{n\times d}\), \(\vec v\in\mathbb R^d\), scalar \(s\))

1. \(M\vec v\): vector in \(\mathbb R^n\) (shape \(n\times1\)).
2. \(\vec vM\): undefined (as written, column vector on left).
3. \(\vec v^2\): undefined in standard linear algebra.
4. \(M^TM\): \(d\times d\) matrix.
5. \(MM^T\): \(n\times n\) matrix.
6. \(\vec v^TM\vec v\): undefined in general (dimension mismatch unless special case \(n=d\)).
7. \((sM\vec v)\cdot(sM\vec v)\): scalar.
8. \((s\vec v^TM^T)^T\): vector in \(\mathbb R^n\) (shape \(n\times1\)).
9. \(\vec v^TM^TM\vec v\): scalar.
10. \(\vec v\vec v^T+M^TM\): \(d\times d\) matrix.
11. \(\frac{M\vec v}{\|\vec v\|}+(\vec v^TM^TM\vec v)M\vec v\): vector in \(\mathbb R^n\), assuming \(\vec v\neq0\).

**Activity 1.3**

\[
A=\begin{bmatrix}2&1\\3&4\\-1&1\end{bmatrix},
\quad
B=\begin{bmatrix}1&0&2\\2&1&3\end{bmatrix},
\quad
C=\begin{bmatrix}1&0&2&-1\\0&1&1&1\\1&1&0&-1\end{bmatrix}.
\]

\[
AB=
\begin{bmatrix}4&1&7\\11&4&18\\1&1&1\end{bmatrix},
\quad
(AB)C=
\begin{bmatrix}11&8&9&-10\\29&22&26&-25\\2&2&3&-1\end{bmatrix}.
\]

\[
BC=
\begin{bmatrix}3&2&2&-3\\5&4&5&-4\end{bmatrix},
\quad
A(BC)=
\begin{bmatrix}11&8&9&-10\\29&22&26&-25\\2&2&3&-1\end{bmatrix}.
\]
Same result by associativity: \((AB)C=A(BC)\).

Transpose formula:
\[
(ABC)^T=C^TB^TA^T.
\]
