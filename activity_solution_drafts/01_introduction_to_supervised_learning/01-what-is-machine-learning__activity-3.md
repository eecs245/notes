---
notebook: 01_introduction_to_supervised_learning/01-what-is-machine-learning.ipynb
activity: 3
cell_index: 18
---

# Activity 3

This is Activity 3.

Source notebook: [`01_introduction_to_supervised_learning/01-what-is-machine-learning.ipynb`](../../01_introduction_to_supervised_learning/01-what-is-machine-learning.ipynb)

## Activity Text

:::{tip} Activity 3
:class: dropdown

Why do you think most of the values above are 0?
:::

## Draft Solution

Most MNIST pixel values are 0 because most pixels are background.

Each image is 28x28, but the handwritten digit occupies only a small subset of pixels. The remaining pixels are black/blank, so their intensity is near 0. In other words, the data is sparse in pixel space.
