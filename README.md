# Introduction
In this project, we want to design a system using natural language processing algorithms that guesses the genre of each film using its synopsis.
The data we use in this project is related to a number of videos that have been downloaded from the IMDB site. Each film has a synopsis, and the web belongs to one or more genres.

## This project has three parts which are as follows:
### Phase1. Use word2vec:
We use the most advanced word2vec algorithm to represent the words in the synopsis. To represent the whole paragraph, we use the average representations of its words. After obtaining this representation, we teach a calculator on the most data that determines the genre or genres of each test data film.
### Phase2. BoW:
In the second part, we want to do the same task before this time with the help of Bag of Words. Here we get the representation of each paragraph in the database using tf-idf, and we teach a calculator on the most data. We compare the accuracy of the calculator on the test data in this section with the accuracy obtained from the previous section and analyze the results.
### Phase3. Improved algorithms:
We propose a way to combine the two previous algorithms that perform better on the data set than either. Similar to the previous two sections, we train our proposed algorithm on the most data and check its accuracy on the data test. We also test different classifiers for all three parts and check the performance of each.


