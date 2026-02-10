---
notebook: 03_vectors/01-vectors-and-linear-combinations.ipynb
activity: 2
cell_index: 15
---

# Activity 2

This is Activity 2.

Source notebook: [`03_vectors/01-vectors-and-linear-combinations.ipynb`](../../03_vectors/01-vectors-and-linear-combinations.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown

In the cell above, try and define $\vec w = \begin{bmatrix} 6 \\ -2 \\ 3 \end{bmatrix}$ as an array and add it to `u`. What error do you see?

:::

## Draft Solution

You should see a shape mismatch/broadcasting error, e.g.

`ValueError: operands could not be broadcast together with shapes (2,) (3,)`

Reason: `u` has 2 components while `w` has 3. Vector addition requires matching dimensions.
