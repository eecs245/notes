# Summation Notation

This course will rely on your understanding of some ideas from calculus and high school algebra. This part of the notes serve to review these ideas, and to introduce some notation that we'll use throughout the course.

Regardless of your background, make sure to skim the content of this section. Core course content starts in [Part 1: Supervised Learning](../01-supervised-learning/01-models-loss.md).

The $\sum$ symbol, read “sigma”, is used to indicate a sum of some sequence. For example, $\displaystyle \sum_{i = 1}^n i^2$ represents the sum of the squares of all integers from 1 to $n$:

$$\sum_{i = 1}^6 i^2 = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2$$

$i$, the index of summation, tells us where to start and end our sum. In general, we have:

$$\sum_{i = a}^b f(i) = f(a) + f(a+1) + f(a+2) + … + f(b-1) + f(b)$$

This is similar to a Python for-loop:

```python
sum = 0
for i in range(a, b+1):
	sum = sum + f(i)
```

In summation notation, the ending index is _inclusive_, unlike in Python.