My goal is to restructure the existing course notes to be divided into more chapters, with some sections (pages) divided into smaller pages, with the overall goal of making the notes more readable and easier to navigate.

Here's the new table of contents I propose. In square brackets, I've added the source pages and the relevant content to be pulled from them. Here are the steps you need to take:

1. Rename each old chapter folder to start with `OLD_`.
1. Create as many new chapter folders as needed, each with a name in the format `03_vectors` (for example).
1. In each chapter folder, create a new file called `00-index.ipynb` that contains the table of contents for the chapter.
1. In each chapter folder, create one `.ipynb` file for each section of the chapter, and fill that `.ipynb` file with the specified content from the old chapter.
1. In cases where an old page was split into multiple new pages, add a transitionary paragraph to the end of the old page and the start of the new page.
1. Update `myst.yml` to reflect the new chapter and section structure.
1. Read through all new pages and make sure all references to old sections/pages are updated (e.g. old references to Chapter 2.7 should now refer to Chapter 5.1, for example), making edits as necessary.

In many cases, you'll need to add a transitionary paragraph to the end of one page and the start of the next, **only do this if an old page was split into multiple new pages**. When you split pages, make sure that the activities in the new pages are always numbered 1, 2, 3, etc.

Ch. 1: Introduction to Supervised Learning
1.1: What is Machine Learning? [Old 1.1]
1.2: Squared Loss and the Constant Model [Old 1.2]
1.3: Absolute Loss [Old 1.3, up until but before "Comparing Loss Functions"]
1.4: Comparing Loss Functions [Old 1.3, "Comparing Loss Functions" to end]

Ch. 2: Simple Linear Regression
2.1: Overview [Old 1.4, up until but before "Functions of Multiple Variables"]
2.2: Detour: Partial Derivatives [Old 1.4, "Functions of Multiple Variables" section]
2.3: Finding Optimal Parameters [Old 1.4, "Minimizing Mean Squared Error" section]
2.4: Correlation [Old 1.4, "Correlation" section]
2.5: Least Squares [Old 1.5]

Ch. 3: Vectors
3.1: Vectors and Linear Combinations [Old 2.1, up until but before "Norms, Revisited"]
3.2: Norms [Old 2.1, "Norms, Revisited" section]
3.3: The Dot Product [Old 2.2]
3.4: Projecting onto a Single Vector [Old 2.3]

Ch. 4: Linear Independence
4.1: Span [Old 2.4, up until but before "Linear Independence"]
4.2: Linear Independence [Old 2.4, "Linear Independence" section]
4.3: Vector Spaces, Basis, and Dimension [Old 2.6]
4.4: Lines, Planes, and Hyperplanes [Old 2.5]

Ch. 5: Matrices
5.1: Matrix Operations [Old 2.7, up until but before "The Transpose"]
5.2: Special Matrices [Old 2.7, from "The Transpose" to end; add a TODO to discuss symmetric, triangular, diagonal, orthogonal, etc.]
5.3: Rank and Column Space [Old 2.8, up until but before "Null Space"]
5.4: Null Space and the Rank-Nullity Theorem [Old 2.8, "Null Space" to end; add a TODO to make more prominent the relationship between the column space, null space, row space, etc.]

Ch. 6: Linear Transformations and Projections
5.5: Linear Transformations [Old 2.9, ONLY the linear transformations section, nothing else]
5.6: Inverses [Old 2.9, everything other than the linear transformations section; keep everything before and after]
5.7: Projecting onto the Column Space [Old 2.10, up until but before "what if $X$'s columns are linearly dependent?"]
5.8: The Complete Solution [Old 2.10, from "what if $X$'s columns are linearly dependent" onwards]

Ch. 7: Regression using Linear Algebra
7.1: Regression using Linear Algebra [Old 3.1]
7.2: Multiple Linear Regression [Old 3.2]

Ch. 8: Gradients
Ch. 8.1: The Gradient Vector [Old 4.1, up until but before "Examples"]
Ch. 8.2: Gradients of Matrix-Vector Operations [Old 4.1, "Examples" onwards]
Ch. 8.3: Gradient Descent [Old 4.2, up until but before "Gradient Descent for Empirical Risk Minimization"]
Ch. 8.4: Gradient Descent for Empirical Risk Minimization [Old 4.3, "Gradient Descent for Empirical Risk Minimization" to end]
Ch. 8.5: Placeholder for Convexity

Ch. 9: Eigenvalues and Eigenvectors
Ch. 9.1: Eigenvalues and Eigenvectors [Old 5.1]
Ch. 9.2: The Rayleigh Quotient [leave as a placeholder for now]
Ch. 9.3: Markov Chains and Adjacency Matrices [Old 5.1 part 2 which is a separate notebook right now]
Ch. 9.4: Multiplicities and Diagonalization [Old 5.2, up until but before "Symmetric Matrices and the Spectral Theorem"]
Ch. 9.5: Symmetric Matrices and the Spectral Theorem [Old 5.2, "Symmetric Matrices and the Spectral Theorem" to end]

Ch. 10: Singular Value Decomposition
Ch. 10.1: Computing the Singular Value Decomposition [Old 5.3, up until but before "Full vs. Compact SVD"]
Ch. 10.2: Low-Rank Approximation [Old 5.3, "Full vs. Compact SVD" to end]
Ch. 10.3: The Best Direction [Old 5.4, up until but before "Principal Components"]
Ch. 10.4: Principal Components Analysis [Old 5.4, "Principal Components" onwards]