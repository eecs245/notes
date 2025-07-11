# Mathematics for Machine Learning

### Course Notes for [EECS 245](https://eecs245.org) at the University of Michigan

Written by Suraj Rampure (rampure@umich.edu; [rampure.org](https://rampure.org))

:::{warning}
These notes are under active development this summer and may contain errors. Check back at the start of the Fall 2025 semester for a more complete version.
:::

Linear algebra, calculus, and probability form the basis of modern machine learning and artificial intelligence. **This course will introduce linear algebra from scratch by focusing on methods and examples from machine learning.** It will give students strong intuition for how linear algebra, calculus and probability are used in machine learning. While the course is primarily theoretical, we’ll look at practical applications involving real data in Python each week, so that students are able to apply what they’ve learned.

There are a plethora of great resources for learning these ideas already, and I've read and taken inspiration from _many_ of them (linked below) in writing these notes and preparing for EECS 245.

**So, why write these notes?** I – like other teachers – have a "story" in my mind that I want to tell, and that story has a particular order and flair that I haven't seen in other books or courses on linear algebra. Most linear algebra courses start with analyzing systems of equations and unknowns. These are extremely important to machine learning, as they are in other fields, but not in an immediately obvious way (at least not to me). This course will start by introducing the foundations of supervised learning – a branch of machine learning that deals with predicting a target variable, $y$, given a set of input variables, $x$ – and dive deep into the relevant ideas from linear algebra as they become necessary to advance this story.

This is, admittedly, an experiment in how to teach linear algebra. We may change directions in future iterations of the course if it turns out that the best practice is to cover content in a more conventional order. But, I'm excited and optimistic about this plan, and you, the student, are sure to learn a great deal regardless.

While I could refer students to different books for different topics, I want to provide a single source of truth that I can refer students to, so that the course narrative is consistent. This is especially important since EECS 245 will be taught _without_ slides. Instead, in each lecture, I will start with a blank whiteboard (on my iPad) and provide high-level overviews of the content. These notes will follow the same story, but with added details, examples, and exercises.

Additionally, I'm using these digital notes as an opportunity to develop interactive visualizations of various concepts in linear algebra, to bring these ideas to life.

Here are some other resources that I've found helpful in writing these notes. Some of these are also linked in particular pages of the notes.

- Gilbert Strang's [lecture videos](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PL49CF3715CB9EF31D&index=1) and [textbook](https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html) on linear algebra. The intuition Strang provides in his lecture videos is legendary. He also has a new book titled [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/#lafe32) which is similar in spirit to these notes.
- 3blue1brown's [Essence of Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWD9XpO3KkyeJezhNY2NbB0LMSV&index=1) series on YouTube is a great resource for developing a visual intuition for linear algebra.

If you have any feedback or suggestions, please don't hesitate to reach out to me at [rampure@umich.edu](mailto:rampure@umich.edu). (If you're a stranger on the internet who has found these notes, I'd love to hear from you too!)

<!-- TODO embed a visualization -->

The contents of this book are licensed for free consumption under the following license: [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).