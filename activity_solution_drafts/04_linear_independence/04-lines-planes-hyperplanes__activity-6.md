---
notebook: 04_linear_independence/04-lines-planes-hyperplanes.ipynb
activity: 6
cell_index: 20
---

# Activity 6

This is Activity 6.

Source notebook: [`04_linear_independence/04-lines-planes-hyperplanes.ipynb`](../../04_linear_independence/04-lines-planes-hyperplanes.ipynb)

## Activity Text

:::{tip} Activity 6
:class: dropdown
Suppose $\vec u, \vec v, \vec w$ are non-zero vectors in $\mathbb{R}^3$. Show that $\vec u, \vec v, \vec w$ are linearly independent if and only if $(\vec u \times \vec v) \cdot \vec w \neq 0$.

:::

## Draft Solution

A standard identity:
\[
(\vec u\times\vec v)\cdot\vec w = \det[\vec u\ \vec v\ \vec w].
\]
The determinant is nonzero iff the three column vectors are linearly independent.

So:
- If \((\vec u\times\vec v)\cdot\vec w\ne 0\), then the determinant is nonzero, so \(\vec u,\vec v,\vec w\) are linearly independent.
- If \(\vec u,\vec v,\vec w\) are linearly independent, determinant is nonzero, so \((\vec u\times\vec v)\cdot\vec w\ne 0\).

Hence
\[
\vec u,\vec v,\vec w\text{ are LI }\iff (\vec u\times\vec v)\cdot\vec w\ne 0.
\]
