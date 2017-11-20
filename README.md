# Machine Learning Engineer Nanodegree
(Projects are listed in reverse chronological order)

## Capstone Project: Kaggle's Personalized Medicine

## Overview
The host, Memorial Sloan Kettering Cancer Center (MSKCC), has been maintaining the database OncoKB for the purpose of knowledge sharing of mutation effects among oncologist. Currently this interpretation of genetic mutations is being done manually. This is a very time-consuming task where a clinical pathologist has to manually review and classify every single genetic mutation based on evidence from text-based clinical literature. And due to my combined interest in biology and data science, I am interested in using my expertise to develop a machine learning algorithm that, using an expert-annotated knowledge base as a baseline, automatically classifies genetic variations.

This is my very first competition since I started my journey of data science the middle of last year. And it has become a unforgetful fast learning experience for me. The rigorious background research, heated forum discussions, sometimes unpredictable model responses, and  the hectic and exhuasting feature engineering implementations, have altoghether given me a read taste of the machine learning implementation on an actual need. The competition ends at October 2nd, 2017. I am very happy to see the ranking of **top 5% among 1300+ teams** in my novice trial as reward of the hardwork I have put. If you are interested, you can check my `doumentation` for more details about the solution.

I found it’s rather interesting to see how machine learning can help in routine biological study. Actually, there are tons of repetitive work in a junior researcher’s career. For me, as my project is using biophysical method to study the structural and functional characteristics of a pathological protein, I am exposed to lots of NMR data every day. It always takes weeks or even months to manually assign amino acid fingerprint son HSQC to its primary structure by comparing their chemical shift with empirical values, which is totally boring and mechanic. Therefore, I am very interested in applying my knowledge and skills I learned here to change this situation.

**Note**
- The mentioned kaggle competition is [here](https://www.kaggle.com/c/msk-redefining-cancer-treatment).
- All the files mentioned could be found in `data` folder or accessed [here](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data).  
- My kaggle profile is [here](https://www.kaggle.com/lu1993).

## Software and Libararies
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org)
- [Gensim (word2vec)](https://radimrehurek.com/gensim/)
- [XGBOOST](https://xgboost.readthedocs.io/en/latest/)
- [Entrez-biopython](http://biopython.org/DIST/docs/api/Bio.Entrez-module.html)

## Skills Used
- Research and investigate a real-world problem of interest.
- Accurately apply specific machine learning algorithms and techniques.
- Properly analyze and visualize your data and results for validity.
- Quickly pick up unfamiliar libraries and techniques.
- Prioritize and implement numerous of ideas and hypothesis.
- Document and write a structed report.

## References

[0] [Kaggle Competition: Personalized Medicine](https://www.kaggle.com/c/msk-redefining-cancer-treatment)
    
[1] [OncoKB: A Precision Oncology Knowledge Base](http://ascopubs.org/doi/full/10.1200/PO.17.00011)

[2] [Cancer-specific High-throughput Annotation of Somatic Mutations: computational prediction of driver missense mutations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2763410/)

[3] [Predicting the Functional Consequences of Somatic Missense Mutations Found in Tumors](https://www.ncbi.nlm.nih.gov/pubmed/24233781)

[4] [Predicting the functional impact of protein mutations: application to cancer genomic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3177186/)

[5] [tmVar: A text mining approach for extracting sequence variants in biomedical literature](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmvar/)

[6] [TaggerOne: Joint Named Entity Recognition and Normalization with Semi-Markov Models](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/taggerone/)

[7] [GNormPlus: An Integrative Approach for Tagging Gene, Gene Family and Protein Domain](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/)

[8] [ Personalised Medicine - EDA with tidy R]( https://www.kaggle.com/headsortails/personalised-medicine-eda-with-tidy-r)

[9] [Redefining Treatment]( https://www.kaggle.com/the1owl/redefining-treatment-0-57456)

[10] [Brief insight on Genetic variations]( https://www.kaggle.com/dextrousjinx/brief-insight-on-genetic-variations)

[11] [Human Genome Variation Society]( http://www.hgvs.org/)

[12] [Official external data and pre-trained models thread]( https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/35810)

[13] [Key Sentences Extraction ideas]( https://github.com/suika-go/suika/blob/master/kernel/baseline.ipynb)

[14] [KAGGLE ENSEMBLING GUIDE]( https://mlwave.com/kaggle-ensembling-guide/)

[15] [Introduction to Ensembling/Stacking in Python]( https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)

[16] [Titanic Top 4% with ensemble modeling]( https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)

[17] [Entrez-biopython]( http://biopython.org/DIST/docs/api/Bio.Entrez-module.html)


## Deep Learning Project: Classifying CIFAR-10 Images

## Overview
In this project, I classified images from the CIFAR-10 dataset, which is consists of airplanes, dogs, cats, and other objects. I first preprocessed the images, then trained a convolutional neural network on all the samples.
After that, I normalized the image features and one-hot encoded the labels. Finally, I applied the concepts and techniques I have learned to build a convolutional, max pooling, dropout, and fully connected layers. At the end, I checked and optimized the neural network's predictions on the sample images.

## Software and Libraries
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org)

## Reinforcement Learning Project: Train a Smartcab How to Drive

## Overview

In this project I applied reinforcement learning techniques for a self-driving agent in a simplified world to aid it in effectively reaching its destinations in the allotted time. I first investigated the environment the agent operates in by constructing a very basic driving implementation. Once my agent was successful at operating within the environment, I proceeded to identify each possible state the agent can be in when considering such things as traffic lights and oncoming traffic at each intersection. With states identified, I implemented a Q-Learning algorithm for the self-driving agent to guide the agent towards its destination within the allotted time. Finally, I improved upon the Q-Learning algorithm to find the best configuration of learning and exploration factors to ensure the self-driving agent is reaching its destinations with consistently positive results.

## Software and Libraries
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [PyGame](http://pygame.org/)

## Unsupervised Learning Project: Creating Customer Segments

## Overview
In this project I applied unsupervised learning techniques on product spending data collected for customers of a wholesale distributor in Lisbon, Portugal to identify customer segments hidden in the data. I first explored the data by selecting a small subset to sample and determine if any product categories highly correlate with one another. Afterwards, I preprocessd the data by scaling each product category and then identifying (and removing) unwanted outliers. With the good, clean customer spending data, I applied PCA transformations to the data and implement clustering algorithms to segment the transformed customer data. Finally, I compared the segmentation found with an additional labeling and consider ways this information could assist the wholesale distributor with future service changes.

## Skills Used
- Apply preprocessing techniques such as feature scaling and outlier detection.
- Interpret data points that have been scaled, transformed, or reduced from PCA.
- Analyze PCA dimensions and construct a new feature space.
- Optimally cluster a set of data to find hidden patterns in a dataset.
- Assess information given by cluster data and use it in a meaningful way.

## Software and Libraries
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)

## Supervised Learning Project: Finding Donors for CharityML

## Overview
In this project, I applied supervised learning techniques and an analytical mind on data collected for the U.S. census to help CharityML (a fictitious charity organization) identify people most likely to donate to their cause. I first explored the data to learn how the census data is recorded. Next, I applied a series of transformations and preprocessing techniques to manipulate the data into a workable format. I then evaluated several supervised learners on the data, and  picked best suited for the solution. Afterwards,  optimized the model as the solution to CharityML. Finally, I explored the chosen model and its predictions under the hood, and I found it performed quite well considering the data it's given.

## Skills Used
- Identify when preprocessing is needed, and how to apply it.
- Establish a benchmark for a solution to the problem.
- Investigate whether a candidate solution model is adequate for the problem.

## Software and Libraries
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)

## Supervised Learning Project: Predicting Boston Housing Prices

## Overview
In this project, I applied supervised machine learning concepts and techniques on data collected for housing prices in the Boston, Massachusetts area to predict the selling price of a new home.I explored the data to obtain important features and descriptive statistics about the dataset, then splitted the data into testing and training subsets, and decided the most suitable performance metric for this problem, and finally built a fairly performed model for this problem.

## Skills Used
- NumPy to investigate the latent features of a dataset.
- Analyze various learning performance plots for variance and bias.
- Determine the best-guess model for predictions from unseen data.
- Evaluate a model's performance on unseen data using previous data.

## Software and Libraries
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)









