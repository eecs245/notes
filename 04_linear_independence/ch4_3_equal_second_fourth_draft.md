# Chapter 4.3 Draft Insert (No Notebook Changes)

Target notebook: `/Users/surajrampure/Desktop/245/notes/04_linear_independence/03-vector-spaces-basis-dimension.ipynb`

Planned heading rename in the **Basis and Dimension** section:
- `### Examples` -> `### Quick Examples`

Planned penultimate section (insert right before `:::{tip} Activity 3`):

### Example: Set of Vectors with Equal Second and Fourth Components

Consider the set

$$S = \left\{ \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \end{bmatrix} \in \mathbb{R}^5 : x_2 = x_4 \right\}$$

In words, $S$ contains all vectors in $\mathbb{R}^5$ whose second and fourth components are equal.

First, let's check that $S$ is a subspace of $\mathbb{R}^5$.

Any vector in $S$ can be written as

$$\begin{bmatrix} a \\ b \\ c \\ b \\ d \end{bmatrix}$$

for some scalars $a, b, c, d$.

- The zero vector is in $S$, since

$$\begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$

has equal second and fourth components.

- If

$$\vec u = \begin{bmatrix} a \\ b \\ c \\ b \\ d \end{bmatrix}, \quad \vec v = \begin{bmatrix} e \\ f \\ g \\ f \\ h \end{bmatrix} \in S,$$

then for any scalars $\alpha, \beta$,

$$\alpha \vec u + \beta \vec v = \begin{bmatrix}
\alpha a + \beta e \\
\alpha b + \beta f \\
\alpha c + \beta g \\
\alpha b + \beta f \\
\alpha d + \beta h
\end{bmatrix}$$

and the second and fourth components are still equal.

So $S$ is a subspace.

Now let's find a basis. Since vectors in $S$ have the form $\begin{bmatrix} a \\ b \\ c \\ b \\ d \end{bmatrix}$,

$$\begin{bmatrix} a \\ b \\ c \\ b \\ d \end{bmatrix}
= a \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
+ b \begin{bmatrix} 0 \\ 1 \\ 0 \\ 1 \\ 0 \end{bmatrix}
+ c \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}
+ d \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}.$$

Therefore,

$$S = \text{span}\left(\left\{
\begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix},
\begin{bmatrix} 0 \\ 1 \\ 0 \\ 1 \\ 0 \end{bmatrix},
\begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix},
\begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}
\right\}\right).$$

These four vectors are linearly independent, so they form a basis for $S$. Hence,

$$\dim(S) = 4.$$
