---
notebook: 00_math_foundations/01-summation.ipynb
activity: 1
cell_index: 0
---

# Activity 1

This is Activity 1.

Source notebook: [`00_math_foundations/01-summation.ipynb`](../../00_math_foundations/01-summation.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown
Identify what makes each of the following expressions invalid or ambiguous, and why.

- $\displaystyle \sum_{k = 1}^{13} k^2 + k$
- $\displaystyle i \sum_{i = 1}^5 i$
- $\displaystyle \sum_{i = 1}^n x_i + x_i$
- $\displaystyle \sum_{i = 5}^3 i^2$
- $\displaystyle \sum_{i = 2}^\pi x_i$

:::

## Draft Solution

- `\sum_{k=1}^{13} k^2 + k` is ambiguous as written. It can mean either `(\sum_{k=1}^{13} k^2) + k` (invalid because `k` is then free/undefined) or `\sum_{k=1}^{13}(k^2+k)`.
- `i\sum_{i=1}^5 i` reuses `i` both as an outside factor and as the summation index. The outside `i` is undefined/ambiguous.
- `\sum_{i=1}^n x_i + x_i` has the same issue: the trailing `x_i` is outside the summation and uses an index that is no longer bound.
- `\sum_{i=5}^3 i^2` has lower limit greater than upper limit. In this course context, that is invalid/undefined unless an "empty sum" convention is explicitly given.
- `\sum_{i=2}^{\pi} x_i` is invalid because summation indices must run over integers, and `\pi` is not an integer bound.
