# Manuscript Datasets

We employ the full initial-structure-to-relaxed-energy (IS2RE) training dataset provided by the OC20 dataset, but nevertheless we include the specific version of the dataset we used.

We also provide the metadata for adsorbate/alloy system subselection, as well as the IS2RE ValID dataset, which we employ as a test dataset.

Using the datasets provided from the Open Catalyst Project repo should work as well, but we provide the specific dataset copies used below:

| Dataset     | Description |
| ----------- | ----------- |
| OC20 Metadata      | [Download](https://www.dropbox.com/s/mmeftwz2vzy39xf/metadata.zip?dl=0)       |
| IS2RE Training   | [Download](https://www.dropbox.com/s/4p4ww2luyawz0gc/IS2RE_training_dataset.zip?dl=0)        |
| IS2RE ValID Test   | [Download](https://www.dropbox.com/s/xktcxkd0yhvn6g0/IS2RE_ValID_test_dataset.zip?dl=0)        |
| Ensemble Training Folds   | [Download](https://www.dropbox.com/s/kokjyxfdh2svx84/5_fold_ensemble_training.zip?dl=0)        |
| H* Adsorbate Datasets   | [Download](https://www.dropbox.com/s/kedowfzt9uqnd83/H_adsorbate_systems.zip?dl=0)        |
| All Datasets   | [Download](https://www.dropbox.com/s/m6zoxg6z45fs49b/datasets.zip?dl=0)        |


# Manuscript Results
___
We provide the results of each UQ method below; that is to say, we provide the (1) adsorption energy predictions and (2) the associated uncertainties of the predictons. We also (3) provide the evidential parameters used to construct the predictive uncertainty for the case of evidential regression.

Each download contains a 'production log', which is what we used to keep track of the experiment. A production log is an inventory of the jobs ran to accomplish the study. Here is a sample of what one might see in a production log:

| Job Description     | Timestamp | Job-ID     | Seed |
| ----------- | ----------- | ----------- | ----------- |
| All IS2RE CGCNN Dropout DR05 Val ID Prediction      |  2022-04-30-20-00-00       |  job-8506782      | 0       |

A description of the job purpose is given (in this case, a CGCNN is trained on the full IS2RE dataset with dropout and dropout is applied on the ValID test dataset with a dropout rate of 5%). The timestamp says when the job was submitted. The Job-ID is the unique SLURM job ID for that job submitted to the cluster. Lastly, seed denotes the global seed used for pseudo-random number generation for the training run.

We rely on seeds for reproducibility and different MC dropout predictions. Note that setting the seed is not guaranteed to make a machine learning training scheme exactly reproducible; rather, it improves the reproducibility of the work by minimizing the amount of unforeseen stochastisity in the training loop.

| Results     | Description |
| ----------- | ----------- |
| Ensemble      | [Download](https://www.dropbox.com/s/1zhkg0d1s5yoicr/ensemble_results.zip?dl=0)       |
| MC Dropout   | [Download](https://www.dropbox.com/s/l6h78ji6hn4ob5b/mc_dropout_results.zip?dl=0)       |
| Evidential Regression   | [Download](https://www.dropbox.com/s/lf0k60u1mfo3wv1/evidential_results.zip?dl=0)       |
| All Results  | [Download](https://www.dropbox.com/s/2iuokiiyfvug3v4/results.zip?dl=0)       |