---
notebook: 05_matrices/01-matrix-operations.ipynb
activity: 4
cell_index: 10
---

# Activity 4

This is Activity 4.

Source notebook: [`05_matrices/01-matrix-operations.ipynb`](../../05_matrices/01-matrix-operations.ipynb)

## Activity Text

:::{tip} Activity 4
:class: dropdown

**Activity 4.1**

Let $P = \begin{bmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$, $S = \begin{bmatrix} 4 & 0 & 0 \\ 0 & \frac{1}{2} & 0 \\ 0 & 0 & 3 \end{bmatrix}$, and $\vec x = \begin{bmatrix} 4 \\ 6 \\ 12 \end{bmatrix}$.

1. Evaluate $P \vec x$ and $S \vec x$. Then, explain in words what multiplying $P$ and $S$ by $\vec x$ does to $\vec x$.
1. Evaluate $PS \vec x$ and $SP \vec x$. The results should be different, as we'd expect, since matrix multiplication is not commutative in general. Explain the difference intuitively, given the "operations" $P$ and $S$ perform on $\vec x$.

$P$ is called a permutation matrix, and $S$ is called a diagonal matrix.

**Activity 4.2**

The famous Fibonacci sequence of integers, $F_0, F_1, F_2, \ldots$, is defined as follows:

$$F_0 = 0, \quad F_1 = 1, \quad F_n = F_{n-1} + F_{n-2} \text{ for } n \geq 2$$

The first few terms in the sequence are $0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, \ldots$.

It turns out you can compute the $n$th term in the sequence using matrix multiplication. 

1. Find a $2 \times 2$ matrix $A$ such that $A \begin{bmatrix} F_{n-1} \\ F_{n-2} \end{bmatrix} = \begin{bmatrix} F_n \\ F_{n-1} \end{bmatrix}$. The answer uses relatively small numbers.
1. Compute $A \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, i.e. the product of $A$ and the vector $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$.
1. Since $A$ is square, we can multiply it by itself. Compute $A^2 \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $A^3 \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

If you continue this process, you'll find that $A^n \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ is a vector containing the $n$th and $(n-1)$th terms in the Fibonacci sequence!

**Activity 4.3**

Using the same matrices $P$ and $S$ from Activity 4.1, compute $(P - S) \vec x$ and $P\vec x - S \vec x$. Are both the results the same? If so, what property of matrix multiplication guarantees this?

Is the result of $(P - S) \vec x$ interpretable, in the same way that the results of $P \vec x$ and $S \vec x$ were in Activity 4.1?

:::

## Draft Solution

**Activity 4.1**

\[
P=\begin{bmatrix}0&0&1\\1&0&0\\0&1&0\end{bmatrix},
\quad
S=\begin{bmatrix}4&0&0\\0&1/2&0\\0&0&3\end{bmatrix},
\quad
\vec x=\begin{bmatrix}4\\6\\12\end{bmatrix}.
\]

- \(P\vec x=\begin{bmatrix}12\\4\\6\end{bmatrix}\): permutes coordinates \((x_1,x_2,x_3)\mapsto(x_3,x_1,x_2)\).
- \(S\vec x=\begin{bmatrix}16\\3\\36\end{bmatrix}\): scales coordinates by \((4,1/2,3)\).

Now compositions:
\[
PS\vec x=P(S\vec x)=\begin{bmatrix}36\\16\\3\end{bmatrix},
\quad
SP\vec x=S(P\vec x)=\begin{bmatrix}48\\2\\18\end{bmatrix}.
\]
Different results because order matters: "scale then permute" is not the same as "permute then scale".

**Activity 4.2**

A matrix that advances Fibonacci pairs is
\[
A=\begin{bmatrix}1&1\\1&0\end{bmatrix},
\quad
A\begin{bmatrix}F_{n-1}\\F_{n-2}\end{bmatrix}
=
\begin{bmatrix}F_n\\F_{n-1}\end{bmatrix}.
\]

\[
A\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}1\\1\end{bmatrix},
\]
\[
A^2\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}2\\1\end{bmatrix},
\quad
A^3\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}3\\2\end{bmatrix}.
\]

**Activity 4.3**

\[
(P-S)\vec x=P\vec x-S\vec x
=\begin{bmatrix}12\\4\\6\end{bmatrix}-\begin{bmatrix}16\\3\\36\end{bmatrix}
=\begin{bmatrix}-4\\1\\-30\end{bmatrix}.
\]
And
\[
P\vec x-S\vec x=\begin{bmatrix}-4\\1\\-30\end{bmatrix}
\]
as well.

They match by distributivity:
\[
(P-S)\vec x=P\vec x-S\vec x.
\]
Interpretation: yes, it is still a linear transformation that combines permutation and scaling effects, but it is not a pure "just permute" or "just scale" operation.
