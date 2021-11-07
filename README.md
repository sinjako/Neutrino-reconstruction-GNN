This repository contains a selection of files created for my Masters thesis project. In this project i reconstructed neutrino events using Graph Neural networks. Model configurations are available in Models.py, where the final model used is called DynedgeEdgepool, called upon with the parameters DynedgeEdgepool(k=[8,8,8,8].

To recreate the results you would also need the database which can be found at HEP cluster as directed in the thesis.

Given a database, results can be reconstructed in the following manner:

1. Edit  graphsaving.py to hold the correct paths for the database and storage of graphs along with desired characteristics of the graph architectures, then run it.
2. Edit Run_model to pull data from the stored graphs and deposit the final results in desired folders, then run it. The other run_model scripts are there to record other parameter choices.
3. Edit Make_comparison_plot.py to pull results from the previous step and deposit the plots in desired folders, then run it. The other make_plot scripts are there to generate plots for angular and probabilistic regression.

The lossfunction.py contains the logcosh loss function and the Von mises fisher sine cosine loss function. Processing.py contains a number of functions necessary to make the multiprocessing work.
