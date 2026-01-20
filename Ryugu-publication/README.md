# README.md

The purpose of this code is to produce PCA related information and figures for the 
_Three early Solar System isotopic reservoirs inferred from measurements of Asteroid Ryugu_ article
published in Nature Astronomy (in preparation).
All materials required to run the code are included in this repository. This repository is archived and won't see any further changes.


## Dependencies
All code was run with Python 3.11.9 and the following libraries:  
* [Docx 1.1.2](https://python-docx.readthedocs.io/en/latest/)  
* [Matplotlib 3.8.4](https://matplotlib.org)  
* [Numpy 1.26.4](https://numpy.org)  
* [Pandas 2.2.2](https://pandas.pydata.org)  
* [Scipy 1.13.1](https://scipy.org)  
* [Scikit-Learn 1.5.0](https://scikit-learn.org/stable/)  

The code is simple and will most likely work with other Python and library versions, but it hasn't been tested on 
anything but what is listed above.

## Usage
There are two python scripts to perform PCA analysis. The first, in the `Default` folder, performs PCA on
carbonaceous and non-carbonaceous samples, but does not include Ti/Rb information. The second, in the `With_TiRb`
folder, performs PCA on only carbonaceous samples, but includes Ti/Rb information. The raw data is included
in each folder, and all preprocessing happens in the script. In this "base" folder, there is a data file called
`Raw_Data_With_Notes.xlsx`. This is a central data file with notes that isn't used by any of the scripts
but includes citations and notes.

The script in `Default` is called `pca_analysis.py`, and the one in `With_TiRb` is called `pca_analysis_TiRb.py`.
The scripts can be run with `python pca_analysis.py` or `python pca_analysis_TiRb.py` while in their 
respective folders. No input arguments are necessary. Other than accounting for the Ti/Rb data differences and 
naming, the files are very similar. The major difference is `Default/pca_analysis.py` writes out the scaled 
data to a `.docx` file that was formatted for inclusion in the associated publication. While both scripts can handle
the included confidence interval data, it should not be used because it is of debatable statistical soundness.

In both scripts, the variable `uc_type` determines whether the standard deviation (SD) or standard error (SE)
uncertainty metric should be used. Furthermore, the variable `iso_drop_list` can be used to drop isotopes
from the analysis, which was used for certain figures in the associated publication. These are the only two locations
that need to be edited to create all published figures. Change the definition of the variable `uc_type` to `2SD`
or `2SE` depending on the uncertainty metric. The `uc_type` variable is defined on line 235 in `pca_analysis.py`
and on line 229 in `pca_analysis_TiRb.py`. Variables can be added at will to the `iso_drop_list` variable that is 
defined on line 240 in `pca_analysis.py` and on line 234 in `pca_analysis_TiRb.py`. The value `"Zn66"` should 
**always** be present in `iso_drop_list`. The plots in the publication are made by dropping some combination
of O17, Ni60, and Ni62. No other modifications need to be made. The scripts will look for the data in the 
same directory they are located in. 

Both scripts save figures to their working directory. The figures have filenames that are dependent on the 
configuration they were run in and are self-explanatory. In general, each script will create a figure
with labels for reference, and a figure without labels that can be labeled as desired. As mentioned above,
`Default/pca_analysis.py` also writes out scaled data to a `.docx` file.

## Citation
If you use this code in your research, please cite:
```
Shollenberger Q.R., Render J., Wimpenny J., Armytage R.M.G., Gunawardena N., Rolison J.M., Simon J.I., Brennecka G.A. (2025) Elemental and isotopic signatures of Asteroid Ryugu support three early Solar System reservoirs. Earth and Planetary Science Letters.
```

## Contributing
This code is used in the journal article cited above, and likely won't be developed much after publication.
However, please feel free to submit any bugfixes or improvements as pull requests and they will be considered.

## License
This project is licensed under the Apache 2.0 License - see the LICENSE.txt and NOTICE.txt files for details.
All new contributions must be made under this license.

## Contact
For questions about the publication, please contact Quinn Shollenberger at shollenberge1@llnl.gov.  
For questions about the code, please contact Nipun Gunawardena at gunawardena1@llnl.gov. 

`LLNL-CODE-2001322`
