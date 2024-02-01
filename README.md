# Modeling and predicting oceanic drifts of plastic waste with physical and data-driven approaches.

## Manon Béchaz

This code is part of a Master's thesis aiming at predicting the oceanic drift of floating objects at the surface of the ocean, using both physical and data-driven approaches. More information on the goal and methods used can be found in the attached report. 

## Stucture of the folders

- ``checkpoints`` : gathers saved checkpoints of the best models obtained for each architecture tested (Models K to P).
- ``configs`` : gathers examples of config files used for training (``configs/config_training.yml``) and for prediction of trajectories (``configs/trace_ISMER_20140629_spot023_drift.yml``).
- ``data_driven`` : gathers the implementations of the data-driven architecture and losses, as well as the training and testing codes.
- ``data_processing`` : gathers all methods used to preprocess the environmental and drift data.
- ``metrics`` : gathers the implementation of the metrics used for evaluation of the predicted trajectories.
- ``models`` : gathers the implementation of the physical module.
- ``notebooks`` : gathers a few (drafts of) notebooks used for different studies. Can be used to reproduce the figures in the report.
- ``utils`` : gatheres a few helpers functions 

## Main code

The file ``main.py`` is provided as an example of how to use the code and can be used to predict trajectories using one of the pretrained models. It requires as an input a config file with paths to environmental data as well as minimum and maximum latitudes and longitudes (an example is provided under ``./configs/trace_ISMER_20140629_spot023_drift.yml``), and the data-driven model to use (letter between K and P).
It outputs and saves a figure with predicted trajectories. 
