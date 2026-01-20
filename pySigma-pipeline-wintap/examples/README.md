# pySigma Wintap Examples

The examples in this directory will show you how to run sigma rules against a sample wintap dataset.

## Running Sigma Rules Against Wintap Dataset

The `wintap_sigma_rules.py` script will format sigma rules in duckdb syntax with wintap fieldnames, and run those rules against a sample of wintap data collect. This script requires the following, additional, python dependencies. Install these into your python environment as necessary (ie `pip install wintappy`)

```
pySigma-backend-duckdb
slugify
tqdm
wintappy
```

This script requires specifying arguments to the location of your dataset and the location of the source sigma rules (for example, where you downloaded the sigma rule repo). Here is an example of running the script:

`python wintap_sigma_rules.py -r ../../sigma/rules/windows/ -d ../../wintap-data-1 -l DEBUG -s`

* Note: the wintap data directory needs to have a processed data set within it, with a `rolling` directory.

### Resources

- [Wintap Dataset](https://gdo-wintap.llnl.gov/) - An open source data set of 'malicious' and 'normal' activity, collected with the wintap sensor.
- [wintappy](https://github.com/LLNL/Wintap-PyUtil) - Python helper utilities for formatting and processing raw wintap data collect
- [Wintap](https://github.com/LLNL/Wintap) - The Wintap Windows event collector
