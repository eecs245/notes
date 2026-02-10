---
notebook: 03_vectors/03-dot-product.ipynb
activity: 6
cell_index: 14
---

# Activity 6

This is Activity 6.

Source notebook: [`03_vectors/03-dot-product.ipynb`](../../03_vectors/03-dot-product.ipynb)

## Activity Text

:::{tip} Activity 6
:class: dropdown

1. What must be true about $\vec u$ and $\vec v$ if equality holds in the Cauchy-Schwarz inequality? That is, if $|\vec u \cdot \vec v| = \|\vec u\| \|\vec v\|$, what relationship do $\vec u$ and $\vec v$ have?

2. What must be true about $\vec u$ and $\vec v$ if equality holds in the triangle inequality? That is, if $\|\vec u + \vec v\| = \|\vec u\| + \|\vec v\|$, what relationship do $\vec u$ and $\vec v$ have? **In what case is there equality for Cauchy-Schwarz but NOT the triangle inequality?**

:::

## Draft Solution

1. Equality in Cauchy-Schwarz,
\[
|\vec u\cdot\vec v|=\|\vec u\|\|\vec v\|,
\]
holds iff \(\vec u\) and \(\vec v\) are linearly dependent (one is a scalar multiple of the other, including opposite direction).

2. Equality in triangle inequality,
\[
\|\vec u+\vec v\|=\|\vec u\|+\|\vec v\|,
\]
holds iff \(\vec u\) and \(\vec v\) point in the same direction (nonnegative scalar multiple).

Case where Cauchy-Schwarz is equality but triangle is not: opposite direction, e.g. \(\vec v=-c\vec u\) with \(c>0\). Then CS is tight, but
\[
\|\vec u+\vec v\|=|1-c|\|\vec u\| < (1+c)\|\vec u\|=\|\vec u\|+\|\vec v\|
\]
unless one vector is zero.
