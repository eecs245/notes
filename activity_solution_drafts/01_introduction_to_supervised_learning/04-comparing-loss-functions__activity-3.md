---
notebook: 01_introduction_to_supervised_learning/04-comparing-loss-functions.ipynb
activity: 3
cell_index: 14
---

# Activity 3

This is Activity 3.

Source notebook: [`01_introduction_to_supervised_learning/04-comparing-loss-functions.ipynb`](../../01_introduction_to_supervised_learning/04-comparing-loss-functions.ipynb)

## Activity Text

:::{tip} Activity 3
:class: dropdown
What is the value of $R_{0,1}(w^*)$ for the constant model $h(x_i) = w$ and 0-1 loss? How does it measure the spread of the data?
:::

## Draft Solution

For 0-1 loss with constant model \(h(x_i)=w\),
\[
R_{0,1}(w)=\frac1n\sum_{i=1}^n \mathbf{1}\{y_i\ne w\}.
\]
To minimize this, choose \(w^*\) as the **mode** (most frequent value). Then
\[
R_{0,1}(w^*) = 1-\frac{\max_v \#\{i:y_i=v\}}{n}.
\]

Interpretation for spread:
- If values are concentrated at one value, this risk is small.
- If values are all distinct, this risk is near 1.

So it measures concentration around the most common value (not full geometric spread like variance).
