---
exports:
  - format: pdf
    template: arxiv_two_column
    output: exports/01-supervised-learning.pdf
---

# Supervised Learning

```{hint} Learning Objectives

```

## Introduction

What is a machine? How does it learn?

## Problem Setup

Consider a dataset of $n$ _scalar_ values, $y_1, y_2, ..., y_n$. The **mean** of $y_1, y_2, ..., y_n$, denoted $\bar{y}$, is given by:

$$
\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i
$$

```{sidebar}
Check it out, we got a sidebar! Let's $f(x) = 5$.
```

```{note}
When you were first taught how to code, the very first program you wrote was likely one that displays `"Hello, world!"`. Similarly, proving that $\sqrt{2}$ is irrational is the standard first example of a **proof by contradiction**.

In a proof by contradiction, we start by assuming the statement we want to prove is **not** true. Since we want to prove that $\sqrt{2}$ is **ir**rational, we'll start by assuming it is **rational**.

If $\sqrt{2}$ is rational, then by the definition of a rational number, it must be possible to write $\sqrt{2} = \frac{p}{q}$ for some integers $p$ and $q$ with $q \neq 0$. Then:

$$\begin{align*} \sqrt{2} &= \frac{p}{q} \\\\ 2 &= \frac{p^2}{q^2} \\\\ 2q^2 &= p^2 \end{align*}$$

What does this last equation tell us? It's telling us that $p^2$ is 2 times some other integer ($q^2$). But, if $p^2$ is 2 times some other integer, then $p^2$ must be even. This tells us that $p$ itself must also be even. 

If $p$ is even, then $p = 2k$ for some integer $k$. Substituting this into the last equation, we get:

$$\begin{align*} 2q^2 &= (2k)^2 \\\\ 2q^2 &= 4k^2 \\\\ q^2 &= 2k^2 \end{align*}$$

This tells us that $q^2$ is 2 times some other integer ($k^2$). But, if $q^2$ is 2 times some other integer, then $q^2$ must be even. This tells us that $q$ itself must also be even.

Why is this a contradiction? Well, we started by assuming that $p$ and $q$ were integers with no common factors. But, if $p$ is even and $q$ is even, then they have a common factor of 2. This contradicts our assumption that $p$ and $q$ have no common factors.

Therefore, our assumption that $\sqrt{2}$ is rational must be false. Therefore, $\sqrt{2}$ is irrational.

```

<!-- 
```{note}
There are other definitions of vectors, too.
- In physics, it's common to define vectors as creatures with **a magnitude and direction**. In one of the exercises later in this page, we'll make sense of this physical definition.
- In mathematics, vectors are often defined as **elements of vector spaces**. What is a vector space? We will soon see.
``` -->