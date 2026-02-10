---
notebook: 00_math_foundations/02-derivatives.ipynb
activity: 2
cell_index: 9
---

# Activity 2

This is Activity 2.

Source notebook: [`00_math_foundations/02-derivatives.ipynb`](../../00_math_foundations/02-derivatives.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown

Note that the activities in this section are quite challenging, so make sure you've attempted and fully understood the examples above first.

**Activity 2.1**

An important function in machine learning is the **sigmoid function**, which is defined as:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$\sigma(x)$ has a nice S-shape, and is used for predicting probabilities.

Find the derivative of $\sigma(x)$, and show that it satisfies the following property:

$$
\frac{\text{d}}{\text{d}x} \sigma(x) = \sigma(x) (1 - \sigma(x))
$$

**Activity 2.2**

Find the derivative of each of the following functions:

- $f(x) = \sqrt{\sin(4\pi x)}$
- $g(x) = (2x + 1)^{3x}$ (Hint: Start by taking the natural logarithm of both sides, then take the derivative of both sides.)

**Activity 2.3**

Suppose $x$ and $y$ satisfy the following relationship:

$$x^2 = y^3 - 11$$

1. Find an expression for $\frac{\text{d}y}{\text{d}x}$ that involves **both** $x$ and $y$. To do this, don't solve for $y$ in terms of $x$ – instead, take the derivative of both sides of the equation with respect to $x$, use the power and chain rules, and re-arrange to isolate $\frac{\text{d}y}{\text{d}x}$.
2. Find the slope of the tangent line to the curve at the point $(x, y) = (4, 3)$.
:::

## Draft Solution

**Activity 2.1 (sigmoid derivative)**

\[
\sigma(x)=\frac{1}{1+e^{-x}}=(1+e^{-x})^{-1}
\]
\[
\sigma'(x)=-(1+e^{-x})^{-2}(-e^{-x})=\frac{e^{-x}}{(1+e^{-x})^2}.
\]
Now
\[
\sigma(x)(1-\sigma(x))
=\frac{1}{1+e^{-x}}\left(1-\frac{1}{1+e^{-x}}\right)
=\frac{e^{-x}}{(1+e^{-x})^2}
=\sigma'(x).
\]

**Activity 2.2 derivatives**

- \(f(x)=\sqrt{\sin(4\pi x)}=(\sin(4\pi x))^{1/2}\)
\[
f'(x)=\frac{1}{2\sqrt{\sin(4\pi x)}}\cdot 4\pi\cos(4\pi x)
=\frac{2\pi\cos(4\pi x)}{\sqrt{\sin(4\pi x)}}
\]
(on the domain where \(\sin(4\pi x)>0\)).

- \(g(x)=(2x+1)^{3x}\). Let \(y=g(x)\). Then
\[
\ln y=3x\ln(2x+1)
\]
\[
\frac{y'}y=3\ln(2x+1)+3x\frac{2}{2x+1}
=3\ln(2x+1)+\frac{6x}{2x+1}.
\]
So
\[
g'(x)=(2x+1)^{3x}\left(3\ln(2x+1)+\frac{6x}{2x+1}\right).
\]

**Activity 2.3 (implicit differentiation)**

Given \(x^2=y^3-11\):
\[
2x=3y^2\frac{dy}{dx}
\quad\Longrightarrow\quad
\frac{dy}{dx}=\frac{2x}{3y^2}.
\]
At \((x,y)=(4,3)\):
\[
\frac{dy}{dx}=\frac{8}{27}.
\]
