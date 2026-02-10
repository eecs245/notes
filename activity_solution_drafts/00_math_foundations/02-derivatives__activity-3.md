---
notebook: 00_math_foundations/02-derivatives.ipynb
activity: 3
cell_index: 14
---

# Activity 3

This is Activity 3.

Source notebook: [`00_math_foundations/02-derivatives.ipynb`](../../00_math_foundations/02-derivatives.ipynb)

## Activity Text

:::{tip} Activity 3
:class: dropdown

Let $q(x) = 8x^4 - 4x$. $q(x)$ has a global minimum at $\left(\frac{1}{2}, -\frac{3}{2}\right)$.

For each of the following functions, find all extrema, and specify whether each extremum is a local maximum, global maximum, local minimum, or global minimum. Make sure to specify both the $x$-values and the $y$-values of each extremum.

1. $f(x) = 2q(x) + 10$
1. $g(x) = -10q(x)$
1. $h(x) = \big| q(x) \big|$
1. Finding the extrema of $l(x) = q(x)^2$ is a bit more complicated than in the examples above. Why?

:::

## Draft Solution

Let \(q(x)=8x^4-4x\), with known global minimum at \(x=\tfrac12\), value \(-\tfrac32\).

1. \(f(x)=2q(x)+10\): positive scaling + shift preserve minimizer.
- Global minimum at \(x=\tfrac12\),
\[
f\!\left(\tfrac12\right)=2\left(-\tfrac32\right)+10=7.
\]
- No maximum (quartic grows to \(+\infty\)).

2. \(g(x)=-10q(x)\): negative scaling flips min to max.
- Global maximum at \(x=\tfrac12\),
\[
g\!\left(\tfrac12\right)=-10\left(-\tfrac32\right)=15.
\]
- No global minimum (goes to \(-\infty\) as \(|x|\to\infty\)).

3. \(h(x)=|q(x)|\):
- Global minima where \(q(x)=0\): \(x=0\) and \(x=\sqrt[3]{\tfrac12}\).
- In the interval where \(q<0\), \(|q|=-q\), so the minimum of \(q\) at \(x=\tfrac12\) becomes a local maximum of \(|q|\):
\[
h\!\left(\tfrac12\right)=\tfrac32.
\]
- No global maximum (\(|q(x)|\to\infty\)).

4. \(\ell(x)=q(x)^2\) is trickier because
\[
\ell'(x)=2q(x)q'(x),
\]
so critical points come from both \(q(x)=0\) and \(q'(x)=0\). You must analyze multiple types of critical points (including repeated-root behavior), not just transformed extrema from a monotone map.
