---
notebook: 00_math_foundations/01-summation.ipynb
activity: 2
cell_index: 1
---

# Activity 2

This is Activity 2.

Source notebook: [`00_math_foundations/01-summation.ipynb`](../../00_math_foundations/01-summation.ipynb)

## Activity Text

:::{tip} Activity 2
:class: dropdown

**Activity 2.1**

Given that $\displaystyle \sum_{i = 1}^n i^2 = \frac{n(n+1)(2n+1)}{6}$, determine the value of $\displaystyle \sum_{j = 3}^{10} (j^2 - 7j)$.

**Activity 2.2**

Show that $\displaystyle \sum_{n = 1}^{100} \frac{1}{n(n+1)} = 1 - \frac{1}{101}$.

_Hint_: Try writing $\frac{1}{n(n+1)}$ as a difference of two fractions.

**Activity 2.3**

Recall, $n! = n \cdot (n-1) \cdot (n-2) \cdot \ldots \cdot 1$.

The Taylor Series expansion for the function $f(x) = e^x$, at the point $x = 0$, is given by:

$$e^x = \sum_{n = 0}^{\infty} \frac{x^n}{n!}$$

Find a closed form expression for the infinite series:

$$\frac{1}{2!} - \frac{1}{3!} + \frac{1}{4!} - \frac{1}{5!} + \frac{1}{6!} - \frac{1}{7!} + \cdots$$

**Activity 2.4**

${n \choose k}$, pronounced "n choose k", is the number of ways to choose $k$ items from a set of $n$ items.

$${n \choose k} = \frac{n!}{k!(n-k)!}$$

For example, ${5 \choose 2} = 10$, because there are 10 ways to choose 2 items from a set of 5 items, and $\frac{5!}{2!(5-2)!} = \frac{120}{2 \cdot 6} = 10$.

Using the fact introduced in Activity 2.1, find a closed form expression for the following sum:

$$\sum_{k = 2}^n {k \choose 2}$$

**Activity 2.5**

Argue why the following equality holds (it'll be hard to prove this algebraically):

$$\sum_{k = 0}^n {n \choose k} = 2^n$$

:::

## Draft Solution

**Activity 2.1**

\[
\sum_{j=3}^{10}(j^2-7j)=\sum_{j=3}^{10}j^2-7\sum_{j=3}^{10}j
\]

\[
\sum_{j=3}^{10}j^2=(1^2+\cdots+10^2)-(1^2+2^2)=385-5=380
\]
\[
\sum_{j=3}^{10}j=(1+\cdots+10)-(1+2)=55-3=52
\]
So
\[
380-7(52)=380-364=16.
\]

**Activity 2.2**

\[
\frac1{n(n+1)}=\frac1n-\frac1{n+1}
\]
so the sum telescopes:
\[
\sum_{n=1}^{100}\frac1{n(n+1)}=\sum_{n=1}^{100}\left(\frac1n-\frac1{n+1}\right)=1-\frac1{101}.
\]

**Activity 2.3**

The series is
\[
\sum_{n=2}^{\infty}\frac{(-1)^n}{n!}.
\]
Using
\[
e^{-1}=\sum_{n=0}^{\infty}\frac{(-1)^n}{n!}=1-1+\sum_{n=2}^{\infty}\frac{(-1)^n}{n!},
\]
we get
\[
\sum_{n=2}^{\infty}\frac{(-1)^n}{n!}=e^{-1}.
\]

**Activity 2.4**

\[
\sum_{k=2}^n\binom{k}{2} = \sum_{k=2}^n \frac{k(k-1)}2
=\frac12\sum_{k=2}^n(k^2-k)
\]
which simplifies to
\[
\frac{n(n+1)(n-1)}6 = \binom{n+1}{3}.
\]

**Activity 2.5**

Combinatorial argument: \(\binom{n}{k}\) counts subsets of size \(k\) from an \(n\)-element set. Summing over all \(k=0,\dots,n\) counts all subsets. The total number of subsets is \(2^n\) (each element is either in or out). Hence
\[
\sum_{k=0}^{n}\binom{n}{k}=2^n.
\]
