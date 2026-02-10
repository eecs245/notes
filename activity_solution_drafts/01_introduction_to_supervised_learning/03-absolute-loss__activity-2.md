---
notebook: 01_introduction_to_supervised_learning/03-absolute-loss.ipynb
activity: 2
cell_index: 18
---

# Activity 2

This is Activity 2.

Source notebook: [`01_introduction_to_supervised_learning/03-absolute-loss.ipynb`](../../01_introduction_to_supervised_learning/03-absolute-loss.ipynb)

## Activity Text

:::{tip} Activity 2 (🎥 walkthrough video!)
:class: dropdown

Suppose we have a dataset of $n = 13$ numbers, such that:

$$0 < y_1 \leq y_2 \leq \ldots \leq y_{13}$$

Given that $y_8 - y_7 > 1$ and $y_9 - y_8 > 1$, how does the value of $R_\text{abs}(y_8 - 1)$ compare to the value of $R_\text{abs}(y_8 + 1)$? Can you determine which is bigger, and by how much?

<iframe width="560" height="315" src="https://www.youtube.com/embed/ibzbb1aO_gI?si=3jfxHIyEvaTmtPgz" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

:::

## Draft Solution

Let \(R_{\text{abs}}(w)=\frac1{13}\sum_{i=1}^{13}|y_i-w|\), with
\[
0<y_1\le\cdots\le y_{13},\quad y_8-y_7>1,\quad y_9-y_8>1.
\]
Set \(a=y_8\). Then \(a-1\in(y_7,y_8)\) and \(a+1\in(y_8,y_9)\).

Compare sums:
\[
\Delta := \sum_{i=1}^{13}|y_i-(a+1)|-\sum_{i=1}^{13}|y_i-(a-1)|.
\]
- For \(i\le 7\): each term increases by 2 (total \(+14\)).
- For \(i=8\): both distances are 1 (change 0).
- For \(i\ge 9\): each term decreases by 2 (5 terms, total \(-10\)).

So \(\Delta=14-10=4\). Therefore
\[
R_{\text{abs}}(a+1)-R_{\text{abs}}(a-1)=\frac{4}{13}>0.
\]
Hence
\[
\boxed{R_{\text{abs}}(y_8+1)=R_{\text{abs}}(y_8-1)+\frac{4}{13}},
\]
so \(R_{\text{abs}}(y_8+1)\) is bigger by \(4/13\).
