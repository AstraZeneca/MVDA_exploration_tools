# MVDA_exploration_tools

Multivariate data analysis (MVDA) exploration tool is a Python library utilizing the scikit-learn library for partial least squares (PLS) and principal components analysis (PCA). <br><br>
Additionally, an example Jupyter notebook shows how to use the pandas library for row and column labelling prior to PCA and PLS modelling.


## Description
One purpose with this code is to name the output loading and score vectors consistent with chemometrics publications and also provide a tools to facilitate plotting of graphs, numbered scatter plots and bar plots based on matplotlib. Another purpose is to deliver a basic example of how to use pandas to keep consistent IDs of rows and columns from the input table(s) to the output results.
### Files
- "PCA_and_PLS_by_pandas_12.ipynb" A jupyter notebook that contains an example run using the pandas library for row and column labelling prior to PCA and PLS modelling.

- "PCA_with_pandas_output.ipynb" The PCA part of the previous file splitted out to yield a shorter example.

- "PLS_with_pandas_output.ipynb" A PLS part of the previous combined file splitted out to yield a shorter example.

- "YO-PLS_with_pandas_output.ipynb" Another PLS part put separately to yield a shorter example.

- "MVDA_exploration_tools.py" A python module containing classes based on scikit-learn that provides traditional chemometrics naming conventions for scores and loading vectors from PCA and PLS modelling.  A class named Fig based on matplotlib is also present to plot modelling results.

- "data/original_spectra.xlsx" contains spectra in the ultra-violett range 250-450 nm with 2 spectral components varied in an experimental design together with concentration values for the training data for optimal modelling with few observations.

- "data/decimated_spectra.xlsx" is the same spectra heavily decimated on the spectral axis in order to facilitate viewing as tables in the above jupyter notebooks.


### Dependencies

* Pandas, and scikit-learn
* Have been tried in Kubuntu 22.04 with Python 3.10 and Windows 10 with Python 3.8 and 3.9, a requirements.txt file exists for the Python 3.10 case.

### Installing

* No installation needed 

### Executing program

* Download the code to a file catalog and run with jupyter lab

## Help

This is an alpha version, please use the Github issues mechanism to resolve issues. 


## Authors

MJosefson, mats.josefson@astrazeneca.com

## Version History


* 0.1
    * Initial Release
* 0.1.2
    * Update to fit changes in scikit-learn v. 1.0
* 0.1.6
    * Updates to handle pandas tables also as output
    * Separated jupyter notebook examples showing pandas output
    * More unit tests implemented
* 0.1.8
    * Update to fit changes in scikit-learn v. 1.3
* 0.1.9
    * Switched PCA from sklearn.decomposition.PCA to use the sklearn.decomposition.randomized_svd directly
    * Corrected PCA and PLS modelling and model properties when variables are unit-variance scaled

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details 

