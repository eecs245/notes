---
notebook: 06_linear_transformations_and_projections/01-linear-transformations.ipynb
activity: 2
cell_index: 28
---

# Activity 2

This is Activity 2.

Source notebook: [`06_linear_transformations_and_projections/01-linear-transformations.ipynb`](../../06_linear_transformations_and_projections/01-linear-transformations.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown

**Activity 2.1**

Find the determinant of the following matrices:

1. $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$
1. $A = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$
1. $A = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}$
1. $A = \begin{bmatrix} 4 & 2 & 0 \\ 3 & 1 & 1 \\ -6 & 2 & 5 \end{bmatrix}$

**Activity 2.2**

Suppose $A$ and $B$ are both $n \times n$ matrices. Is $\text{det}(A + B) = \text{det}(A) + \text{det}(B)$ in general?

**Activity 2.3**

Suppose we multiply $A$'s 2nd column by 3. What happens to $\text{det}(A)$?

**Activity 2.4**

If $A$'s columns are linearly dependent, then find $\text{det}(AB)$.

**Activity 2.5**

1. Find the determinant of $R(\theta) = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}$ (Hint: The answer does not depend on $\theta$!).
1. $R(\theta)$ is a $2 \times 2$ orthogonal matrix. If $Q$ is an $n \times n$ orthogonal matrix, then what is $\text{det}(Q)$?

:::

## Draft Solution

**Activity 2.1 (determinants)**

1. \(\det\begin{bmatrix}1&2\\3&4\end{bmatrix}=1\cdot4-2\cdot3=-2\).
2. \(\det\begin{bmatrix}1&3\\2&4\end{bmatrix}=1\cdot4-3\cdot2=-2\).
3. \(\det\begin{bmatrix}2&1\\4&3\end{bmatrix}=2\cdot3-1\cdot4=2\).
4. \(\det\begin{bmatrix}4&2&0\\3&1&1\\-6&2&5\end{bmatrix}=-30\).

**Activity 2.2**

In general, no:
\[
\det(A+B)\ne \det(A)+\det(B).
\]
Counterexample: \(A=B=I_2\).
\[
\det(A+B)=\det(2I_2)=4,
\quad
\det(A)+\det(B)=1+1=2.
\]

**Activity 2.3**

Multiplying one column by 3 multiplies determinant by 3.

**Activity 2.4**

If columns of \(A\) are linearly dependent, \(\det(A)=0\). Then
\[
\det(AB)=\det(A)\det(B)=0.
\]

**Activity 2.5**

1. For
\[
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix},
\]
\[
\det(R(\theta))=\cos^2\theta+\sin^2\theta=1.
\]

2. If \(Q\) is orthogonal, \(Q^TQ=I\). Taking determinants:
\[
\det(Q^TQ)=\det(Q)^2=1
\Rightarrow \det(Q)=\pm1.
\]
