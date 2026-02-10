---
notebook: 03_vectors/03-dot-product.ipynb
activity: 1
cell_index: 0
---

# Activity 1

This is Activity 1.

Source notebook: [`03_vectors/03-dot-product.ipynb`](../../03_vectors/03-dot-product.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown

**Activity 1.1**

Let $\vec z = \begin{bmatrix} 5 \\ 3 \\ -1 \end{bmatrix}$ and $\vec 1 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$.

1. Find $\vec z \cdot \vec 1$.
1. In general, if $z \in \mathbb{R}^n$ is any vector, and $\vec 1 = \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix}$ is a vector of all 1s with the same number of components as $\vec z$, what is the value of:

$$\vec z \cdot \vec 1$$

**Activity 1.2**

Dot products are useful for computing weighted averages. Let's illustrate that here. In your freshman fall semester, you took the following courses and earned the following grades:

| Course | Grade | Credits | 
| --- | --- | --- |
| EECS 245 | 4 (A+) | 4 |
| MATH 116 | 3.7 (A-) | 3 |
| EECS 201 | 0 (F) | 1 |
| DATASCI 101 | 3.3 (B+) | 4 |

Find your GPA for the semester, and **express it as a dot product** between a grades vector $\vec g$ and a weights vector $\vec w$.

:::

## Draft Solution

**Activity 1.1**

\[
\vec z\cdot\vec 1 = 5+3-1=7.
\]
In general, for \(\vec z=[z_1,\dots,z_n]^T\) and \(\vec 1=[1,\dots,1]^T\):
\[
\vec z\cdot\vec 1 = \sum_{i=1}^n z_i,
\]
so dotting with \(\vec 1\) sums components.

**Activity 1.2**

Semester GPA is a weighted average by credits.

Let
\[
\vec g=\begin{bmatrix}4\\3.7\\0\\3.3\end{bmatrix},
\quad
\vec w=\begin{bmatrix}4/12\\3/12\\1/12\\4/12\end{bmatrix}.
\]
Then
\[
\text{GPA}=\vec g\cdot\vec w
=\frac{4\cdot4+3.7\cdot3+0\cdot1+3.3\cdot4}{12}
=\frac{40.3}{12}\approx 3.3583.
\]
