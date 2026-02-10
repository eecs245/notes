---
notebook: 04_linear_independence/04-lines-planes-hyperplanes.ipynb
activity: 9
cell_index: 20
---

# Activity 9

This is Activity 9.

Source notebook: [`04_linear_independence/04-lines-planes-hyperplanes.ipynb`](../../04_linear_independence/04-lines-planes-hyperplanes.ipynb)

## Activity Text

:::{tip} Activity 9
:class: dropdown
Prove that if you pick any two points on a plane in $\mathbb{R}^n$, the line connecting the two points is contained entirely on the plane.

_Hint: Start by picking two points on the plane. Both of them must satisfy the parametric equation above, just with different values of $s$ and $t$. Then, using what you've learned about parametric equations of lines, find the equation of the line connecting the two. What do you notice about that line?_
:::

## Draft Solution

Let the plane be
\[
P=\{\vec p_0+s\vec u+t\vec v: s,t\in\mathbb R\}.
\]
Pick two points on the plane:
\[
\vec p_1=\vec p_0+s_1\vec u+t_1\vec v,
\quad
\vec p_2=\vec p_0+s_2\vec u+t_2\vec v.
\]
The line through them is
\[
\vec \ell(\lambda)=\vec p_1+\lambda(\vec p_2-\vec p_1),\quad \lambda\in\mathbb R.
\]
Now
\[
\vec p_2-\vec p_1=(s_2-s_1)\vec u+(t_2-t_1)\vec v,
\]
so
\[
\vec \ell(\lambda)=\vec p_0+[s_1+\lambda(s_2-s_1)]\vec u+[t_1+\lambda(t_2-t_1)]\vec v.
\]
This is again of the form \(\vec p_0+s\vec u+t\vec v\), so \(\vec \ell(\lambda)\in P\) for all \(\lambda\).
Therefore the entire connecting line lies in the plane.
