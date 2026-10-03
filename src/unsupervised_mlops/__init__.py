"""Label-efficient learning with unsupervised methods.

Three parts, following Geron, *Hands-On Machine Learning* (2nd ed.), Chapter 9:

* ``clustering_features`` - K-Means as a feature-engineering step for Logistic Regression.
* ``semi_supervised``     - label only cluster-representative images, then propagate labels.
* ``anomaly``             - density-based anomaly detection with Gaussian Mixture Models
                            (BIC/AIC model selection), benchmarked against DBSCAN.
"""

__version__ = "1.0.0"
