# Feature selection in high dimensional data using supervised learning methods

This is a feature selection pipeline. Feature selection is performed using the RFECV wrapper based algorithm for three classifiers which are based on gradient boosting (XgBoost, CatBoost and LightGBM). Then the feature selection results are combined into a consensus list of features through the borda rank based count algorithm. The pipeline has been tested on a binary classification problem with high dimensional scRNA-seq data (cells have to be classified either as healthy or as infected from covid-19).

# How to run

- Clone this project.

- Inside the project's root folder create a folder called datasets.

- Download the example dataset [here](https://www.ebi.ac.uk/gxa/sc/experiment/E-MTAB-9221/download/zip?fileType=normalised&accessKey=) and extract it in the project's datasets folder.

- Download the example dataset tags [here](https://www.ebi.ac.uk/gxa/sc/experiment/E-MTAB-9221/download?fileType=experiment-design&accessKey=) and place it in the project's datasets folder.

- Create a virtual environment and run:

  `pip install -r requirements.txt`

- Preprocess dataset & tags through the data_preprocessor notebook.

- Run the feature selection pipeline through the feature_selector notebook.

- Test out the performance of some widely used classifiers on the original dataset and the reduced one through the cross_val_scores&box_plots notebook.

# Special Thanks To

- Aristidis Vrahatis: Assistant professor, Dept. of Informatics, Ionian University, Corfu, Greece.
