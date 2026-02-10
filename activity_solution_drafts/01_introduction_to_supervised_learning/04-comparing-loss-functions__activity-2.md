---
notebook: 01_introduction_to_supervised_learning/04-comparing-loss-functions.ipynb
activity: 2
cell_index: 14
---

# Activity 2

This is Activity 2.

Source notebook: [`01_introduction_to_supervised_learning/04-comparing-loss-functions.ipynb`](../../01_introduction_to_supervised_learning/04-comparing-loss-functions.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown


Consider the dataset:

$$1 \qquad 3 \qquad 5 \qquad 7 \qquad 64$$

Compute the following:

1. The variance.
1. The mean squared error of the **median**.
1. The mean absolute deviation from the **median**.
1. The mean absolute deviation from the **mean**.

What do you notice about the results to (1) and (2)? What about the results to (3) and (4)?

:::

## Draft Solution

Data: \(1,3,5,7,64\), \(n=5\).

- Mean: \(\bar y=16\).
- Median: \(m=5\).

1. Variance (population):
\[
\sigma^2=\frac{(1-16)^2+(3-16)^2+(5-16)^2+(7-16)^2+(64-16)^2}{5}
=\frac{225+169+121+81+2304}{5}=\frac{2900}{5}=580.
\]

2. MSE of the median:
\[
\frac{(1-5)^2+(3-5)^2+(5-5)^2+(7-5)^2+(64-5)^2}{5}
=\frac{16+4+0+4+3481}{5}=\frac{3505}{5}=701.
\]

3. Mean absolute deviation from the median:
\[
\frac{|1-5|+|3-5|+|5-5|+|7-5|+|64-5|}{5}
=\frac{4+2+0+2+59}{5}=\frac{67}{5}=13.4.
\]

4. Mean absolute deviation from the mean:
\[
\frac{|1-16|+|3-16|+|5-16|+|7-16|+|64-16|}{5}
=\frac{15+13+11+9+48}{5}=\frac{96}{5}=19.2.
\]

What to notice:
- (1) vs (2): variance (MSE of the mean) is smaller than MSE of the median, consistent with mean minimizing squared loss.
- (3) vs (4): MAD from median is smaller than MAD from mean, consistent with median minimizing absolute loss.
