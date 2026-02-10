---
notebook: 10_singular_value_decomposition/03-best-direction.ipynb
activity: 2
cell_index: 14
---

# Activity 2

This is Activity 2.

Source notebook: [`10_singular_value_decomposition/03-best-direction.ipynb`](../../10_singular_value_decomposition/03-best-direction.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown
Suppose $\tilde X$ is a centered $n \times d$ matrix, meaning the mean of each column of $\tilde X$ is 0.

Prove that the entries of $\tilde X \vec w$, where $\vec w$ is any vector in $\mathbb{R}^d$, sum to 0. _Hint: Think about how you can use the vector $\vec 1$._
:::

## Draft Solution

Let \(\vec 1\in\mathbb R^n\) be the all-ones vector.

"Centered columns" means each column sum is 0, equivalently
\[
\vec 1^T\tilde X = \vec 0^T.
\]
For any \(\vec w\in\mathbb R^d\):
\[
\text{sum of entries of }\tilde X\vec w
=\vec 1^T(\tilde X\vec w)
=(\vec 1^T\tilde X)\vec w
=\vec 0^T\vec w
=0.
\]
Hence the entries of \(\tilde X\vec w\) always sum to 0.
