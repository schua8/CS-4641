CS4641 - HW3 - Unsupervised Learning
Sean Chua

All analysis was done using the latest stable version of the Waikato Environment for Knowledge Analysis (WEKA), provided for free by the University of Waikato.

Installation:
1. Install Java 8 JDK
2. Install the latest stable version of WEKA
3. Open WEKA GUI and Explorer
4. Run experiments

WEKA GUI:

Explorer Module(Neural Net Training and Testing):
1. Click on the Explorer Module
2a. Under the Preprocess tab, click on open file to load whatever training file needs to be trained on.
2b. If doing DR or clustering, add necessary filter
3. Click on the Classify tab.
4. Choose: classifiers > functions> multilayerPerceptron
5. Cross validation 10-fold
6. Click start
7. Save data once run is complete

Explorer Module(k-Means/EM clustering):
1. Click on the Explorer Module
2. Under the Preprocess tab, click on open file to load whatever training file needs to be trained on.
3. Click on the Cluster tab.
4. Choose the cluster algorithm to run on the training data.
5. training split of 70%
6. Choose Classes to clusters evaluation for checking classes to cluster error.
7. Select the Ignore attributes and choose the classification attribute so that the clustering is evaluated on that attribute.
8. Click start
9. Save data once run is complete

Explorer Module(PCA/Information Gain Evaluation):
1. Click on the Explorer Module
2. Under the Preprocess tab, click on open file to load whatever training file needs to be trained on.
3. Click on the Select Attributes tab.
4. Choose the Attribute Evaluator dimensionality algorithm to run on the training data.
5. Choose the Search Method to be Ranker.
6. Choose Attribute Selection Mode of full training set.
7. Click start
8. Save data or results once run is complete

Explorer Module(Adjusting dataset or running dimensionality reduction algorithm filters):
1. Click on the Explorer Module
2. Under the Preprocess tab, click on open file to load whatever training file needs to be trained on.
3. Choose Filter (Resampling, Reordering, PCA, ICA, RP, IGAE) that should be run.
4. Click Apply.
5. Save data once filtering is done.
