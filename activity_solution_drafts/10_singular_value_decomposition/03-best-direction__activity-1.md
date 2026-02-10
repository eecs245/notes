---
notebook: 10_singular_value_decomposition/03-best-direction.ipynb
activity: 1
cell_index: 10
---

# Activity 1

This is Activity 1.

Source notebook: [`10_singular_value_decomposition/03-best-direction.ipynb`](../../10_singular_value_decomposition/03-best-direction.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown
Consider the line $\frac{3}{5}x + \frac{4}{5}y = 0$ in $\mathbb{R}^2$. For the point $(1, 1)$,
1. What is the vertical error of projecting $(1, 1)$ onto the line?
2. What is the point on the line that is closest to $(1, 1)$?
3. What is the orthogonal error of projecting $(1, 1)$ onto the line?
:::

## Draft Solution

Line:
\[
\frac35x+\frac45y=0
\quad\Longleftrightarrow\quad
y=-\frac34x.
\]
Point: \((1,1)\).

1. Vertical projection keeps \(x=1\), so projected point is \((1,-3/4)\).
Vertical error (residual) is
\[
1-\left(-\frac34\right)=\frac74,
\]
so magnitude is \(\frac74\).

2. Closest point (orthogonal projection) onto the line through origin.
A direction vector is \(d=(4,-3)\), with unit vector \(\hat d=(4/5,-3/5)\).
\[
(1,1)\cdot\hat d=\frac15,
\quad
p=((1,1)\cdot\hat d)\hat d
=\frac15\left(\frac45,-\frac35\right)
=\left(\frac{4}{25},-\frac{3}{25}\right).
\]
So closest point is
\[
\boxed{\left(\frac{4}{25},-\frac{3}{25}\right)}.
\]

3. Orthogonal error (distance to line):
\[
\left\|(1,1)-\left(\frac{4}{25},-\frac{3}{25}\right)\right\|=\frac75.
\]
So orthogonal error magnitude is \(\boxed{7/5}\).
