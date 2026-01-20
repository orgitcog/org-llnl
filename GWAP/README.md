# GWAP: Ground Water Age Predictor, Version 1.0

Author(s): Abdullah Azhar, Indrasis Chakraborty, Ate Visser, Yang Liu, Jory Chapin Lerback, Erik Oerter


## Installation
- Python Env
    - First  `pip install -r requirements.txt`
    - Then pip install -e 
    

## Project Detail
- Groundwater ages provide insight into recharge rates, flow velocities, and vulnerability to contaminants. The ability to predict groundwater ages based on more accessible parameters via Machine Learning (ML) would advance our ability to guide sustainable management of groundwater resources. In this work, ML models were trained and tested on a large dataset of tritium concentrations (n=2410) and tritium-helium groundwater ages (n=1157) from the California Central Valley, a large groundwater basin with complex land use, irrigation, and water management practices. The dataset collected by the California Waterboards for the Groundwater Ambient Monitoring and Assessment (GAMA) program and can be downloaded from https://gamagroundwater.waterboards.ca.gov/gama/gamamap/public/.


## Code Structure
- The main code is written in Jupyter notebook, main_notebook.ipynb
- Data preparation and augmentation codes: Data_Augmentation.py, data_import_preparation.py, data_prep_imputation_normalizing.py
- Regressor and Classifier codes: decision_tree_regressor.py, decision_tree_classifier.py
- Postprocessing codes: partial_dependence.py, subplots_script.py, z_score_norm.py

### CP NUMBER: CP02868
