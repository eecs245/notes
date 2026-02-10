---
notebook: 00_math_foundations/02-derivatives.ipynb
activity: 5
cell_index: 23
---

# Activity 5

This is Activity 5.

Source notebook: [`00_math_foundations/02-derivatives.ipynb`](../../00_math_foundations/02-derivatives.ipynb)

## Activity Text

:::{tip} Activity 5
:class: dropdown

**Activity 5.1**

Let $f(x) = x \log(x^2)$, where $\log( \cdot )$ is the natural logarithm.

1. Find the critical points of $f(x)$, and determine whether they are local maxima, minima, or neither.
2. Find the inflection points of $f(x)$, and use them to sketch a possible graph of $f(x)$.

**Activity 5.2**

Let $g(3) = 10$, $\frac{\text{d}g}{\text{d}x}(3) = -2$, and $\frac{\text{d}^2g}{\text{d}x^2}(3) = 1$.

1. Describe the behavior of $g(x)$ near $x = 3$.
2. The Taylor series of a function allows us to approximate the value of a function near a point $x = a$, given the value of the function and its derivatives at $x = a$. The Taylor series of an arbitrary function $f(x)$ around $x = a$ is given by:
$$
f(x) \approx f(a) + \left(\frac{\text{d}f}{\text{d}x}(a) \right)(x - a) + \frac{1}{2}\left(\frac{\text{d}^2f}{\text{d}x^2}(a) \right)(x - a)^2 + \frac{1}{6} \left(\frac{\text{d}^3f}{\text{d}x^3}(a) \right)(x - a)^3 + \frac{1}{24} \left(\frac{\text{d}^4f}{\text{d}x^4}(a) \right)(x - a)^4 + \cdots
$$

Note that this is an infinite series; the more terms we use, the more accurate the approximation.

Use the Taylor series to approximate the value of $g(3.1)$, using only the information provided. You'll only be able to use the first 3 terms of the Taylor series.

**Activity 5.3**

Given that $\frac{\text{d}^2h}{\text{d}x^2} = 2x(x - 3)(x + 1)$, sketch a possible graph of $h(x)$.

## Draft Solution

**Activity 5.1**

\[
f(x)=x\log(x^2)=2x\ln|x|,\quad x\neq 0.
\]
\[
f'(x)=2\ln|x|+2,
\]
so critical points satisfy \(\ln|x|=-1\Rightarrow |x|=e^{-1}\), i.e.
\[
x=\pm\frac1e.
\]
Second derivative:
\[
f''(x)=\frac{2}{x}.
\]
- At \(x=\frac1e\), \(f''>0\): local minimum, value \(f(1/e)=-2/e\).
- At \(x=-\frac1e\), \(f''<0\): local maximum, value \(f(-1/e)=2/e\).

Concavity: down on \(( -\infty,0 )\), up on \((0,\infty)\). Concavity changes across 0, but \(x=0\) is not in the domain, so there is no inflection *point* on the graph.

**Activity 5.2**

Given \(g(3)=10\), \(g'(3)=-2\), \(g''(3)=1\):
- Near \(x=3\), the function is decreasing (negative slope) and concave up.

Taylor approximation at \(a=3\):
\[
g(3.1)\approx g(3)+g'(3)(0.1)+\frac12 g''(3)(0.1)^2
=10-0.2+0.005=9.805.
\]

**Activity 5.3**

\[
h''(x)=2x(x-3)(x+1).
\]
Sign chart:
- \(( -\infty,-1 )\): negative (concave down)
- \(( -1,0 )\): positive (concave up)
- \(( 0,3 )\): negative (concave down)
- \(( 3,\infty )\): positive (concave up)

So inflection points occur at \(x=-1,0,3\), with concavity alternating down/up/down/up. Any sketch with that concavity pattern is valid.
