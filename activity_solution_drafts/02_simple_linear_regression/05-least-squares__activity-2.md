---
notebook: 02_simple_linear_regression/05-least-squares.ipynb
activity: 2
cell_index: 6
---

# Activity 2

This is Activity 2.

Source notebook: [`02_simple_linear_regression/05-least-squares.ipynb`](../../02_simple_linear_regression/05-least-squares.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown
I glossed over something important above. I said that minimizing mean squared error is equivalent to minimizing the sum of squared errors.

Why is that true? Let's consider a particular example, of finding $w_0^*$ and $w_1^*$ for the simple linear regression model $h(x_i) = w_0 + w_1 x_i$.

Why are the $w_0^*$ and $w_1^*$ that minimize

$$R_\text{sq}(w_0, w_1) = \frac{1}{n} \sum_{i=1}^n (y_i - (w_0 + w_1 x_i))^2$$

the same as the $w_0^*$ and $w_1^*$ that minimize

$$S_\text{sq}(w_0, w_1) = \sum_{i=1}^n (y_i - (w_0 + w_1 x_i))^2$$

This idea was also reinforced in Lab 1, and in [Activity 3 of Appendix 2](../00_math_foundations/02-derivatives.ipynb#maxima-and-minima).

:::

## Draft Solution

Because
\[
R_{\text{sq}}(w_0,w_1)=\frac1n S_{\text{sq}}(w_0,w_1),
\]
where \(n>0\) is constant, minimizing one is equivalent to minimizing the other.

Formally, for any two parameter pairs \(\theta_a,\theta_b\):
\[
S_{\text{sq}}(\theta_a) \le S_{\text{sq}}(\theta_b)
\iff \frac1n S_{\text{sq}}(\theta_a) \le \frac1n S_{\text{sq}}(\theta_b).
\]
Multiplying by a positive constant does not change ordering, so argmin is unchanged:
\[
\arg\min R_{\text{sq}} = \arg\min S_{\text{sq}}.
\]
