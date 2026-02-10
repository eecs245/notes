---
notebook: 04_linear_independence/03-vector-spaces-basis-dimension.ipynb
activity: 2
cell_index: 1
---

# Activity 2

This is Activity 2.

Source notebook: [`04_linear_independence/03-vector-spaces-basis-dimension.ipynb`](../../04_linear_independence/03-vector-spaces-basis-dimension.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown
1. Suppose we _try_ to create a new vector space in which vector addition is defined in the following way:

$${\color{orange}\begin{bmatrix} u_1 \\ u_2 \end{bmatrix}} + {\color{#3d81f6} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}} = \begin{bmatrix} {\color{orange}u_1} + {\color{#3d81f6}v_2} \\ {\color{orange}u_2} + {\color{#3d81f6}v_1} \end{bmatrix}$$

So for example $\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 1 + 4 \\ 2 + 3 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix}$.

Suppose we stick with the usual definition of scalar multiplication. Which of the 8 conditions above are not satisfied?

2. Suppose we consider the set of vectors with two components, where scalar multiplication is defined as

$$k {\color{#3d81f6}\begin{bmatrix} v_1 \\ v_2 \end{bmatrix}} = \begin{bmatrix} k {\color{#3d81f6}v_1} \\ {\color{#3d81f6}v_2} \end{bmatrix}$$

Suppose we stick with the usual definition of vector addition. Which of the 8 conditions above are not satisfied?
:::

## Draft Solution

I will refer to standard vector-space axioms by name.

1. With modified addition
\[
\begin{bmatrix}u_1\\u_2\end{bmatrix}\oplus
\begin{bmatrix}v_1\\v_2\end{bmatrix}
=
\begin{bmatrix}u_1+v_2\\u_2+v_1\end{bmatrix}
\]
and usual scalar multiplication:
- Closure: holds.
- Commutativity of addition: **fails** in general.
- Associativity of addition: **fails**.
- Additive identity/inverse behavior is not consistent on both sides because of noncommutativity.
- Distributivity over scalar addition, \((a+b)u=au\oplus bu\): **fails**.

So this is not a vector space.

2. With usual addition but modified scalar multiplication
\[
k\odot\begin{bmatrix}v_1\\v_2\end{bmatrix}=\begin{bmatrix}kv_1\\v_2\end{bmatrix}:
\]
- \((ab)\odot v=a\odot(b\odot v)\): holds.
- \(a\odot(u+v)=a\odot u+a\odot v\): holds.
- \((a+b)\odot v=a\odot v+b\odot v\): **fails**.
- Also \(0\odot v\neq 0\) in general (second component stays \(v_2\)).

So this is also not a vector space.
