---
notebook: 00_math_foundations/02-derivatives.ipynb
activity: 1
cell_index: 5
---

# Activity 1

This is Activity 1.

Source notebook: [`00_math_foundations/02-derivatives.ipynb`](../../00_math_foundations/02-derivatives.ipynb)

## Activity Text

:::{tip} Activity 1
:class: dropdown
Let $f(x) = \frac{1}{4}x^4 + \frac{1}{3}x^3 - x^2 + 2$.

Given that $f'(x)=x^3+x^2-2x$, find the equation of the tangent line to $f(x)$ at the following points:

- $x=-3$
- $x=-1$
- $x=1$

As a bonus exercise, try to verify that the provided formula for $f'(x)$ is correct, using your knowledge of derivatives from Calculus 1.
:::

## Draft Solution

Given
\[
f(x)=\frac14x^4+\frac13x^3-x^2+2,\quad f'(x)=x^3+x^2-2x.
\]

- At \(x=-3\):
\[
f(-3)=\frac{17}{4},\quad f'(-3)=-12.
\]
Tangent line:
\[
y-\frac{17}{4}=-12(x+3)
\quad\Longleftrightarrow\quad
y=-12x-\frac{127}{4}.
\]

- At \(x=-1\):
\[
f(-1)=\frac{11}{12},\quad f'(-1)=2.
\]
Tangent line:
\[
y-\frac{11}{12}=2(x+1)
\quad\Longleftrightarrow\quad
y=2x+\frac{35}{12}.
\]

- At \(x=1\):
\[
f(1)=\frac{19}{12},\quad f'(1)=0.
\]
Tangent line:
\[
y=\frac{19}{12}.
\]

Bonus check:
\[
\frac{d}{dx}\left(\frac14x^4\right)=x^3,\quad
\frac{d}{dx}\left(\frac13x^3\right)=x^2,\quad
\frac{d}{dx}(-x^2)=-2x,
\]
so \(f'(x)=x^3+x^2-2x\) is correct.
