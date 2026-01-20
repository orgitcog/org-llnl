<img align="left" width="75" height="75" src="./logo.png"> <br> 
# IBIS

LLNL's Interactive Bayesian Inference and Sensitivity, or IBIS, provides tools to enable understanding how the output of a model is impacted by uncertainty in its inputs. With the sensitivity module we can rank and visualize the effect of inputs on the computational model output. The mcmc module enables us to incorporate experimental data into our understanding of input and output distributions with Bayesian inference.

Sensitivity studies
A variety of methods are availble in IBIS to understand and compare the impacts of model inputs.
   - F-test: f_score
   - mutual information score: mutual_info_score
   - Lasso regression: lasso
   - sensitivity with "spaghetti plot": sensitivity
   - One-at-a-time study: including method of Morris
   - Variance based sensitivity analysis: Sobol indices
   - polynomial chaos expansion: pce_score

   Associated modules
      - sensitivity
      - pce_model
      - plots

Bayesian Inference/Calibration
A Bayesian posterior distribution can be estimated with IBIS's mcmc module. It can estimate input parameter uncertainty and model discrepancy. For more details refer to [Bayesian Calibration of Computer Models](https://www.researchgate.net/publication/4772045_Bayesian_Calibration_of_Computer_Models).
   - default mcmc: Metropolis-Hastings algorithm and multivariate normal likelihood
   - discrepancy mcmc: We can additionally estimate "tau" model discrepancy parameters

   Associated modules
      - mcmc
      - mcmc_diagnostics

Kosh operators
If users have data in a Kosh store, it's convenient to use IBIS's Kosh operators to run the various sensitivity methods or create the mcmc object. The following operators are available:
   - KoshMCMC
   - KoshOneAtATimeEffects
   - KoshSensitivityPlots

## Getting Started

To get the latest public version:

```bash
pip install llnl-ibis
```

To get the latest stable from a cloned repo, simply run:

```bash
pip install .
```

Alternatively, add the path to this repo to your PYTHONPATH environment variable or in your code with:

```bash
import sys
sys.path.append(path_to_ibis_repo)
```
## Documentation
The documentation can be built from the `docs` directory using:

```bash
make html
```

Read the Docs coming soon.

## Contact Info

IBIS maintainer can be reached at: olson59@llnl.gov

## Contributing

Contributing to IBIS is relatively easy. Just send us a pull request. When you send your request, make develop the destination branch on the IBIS repository.

Your PR must pass IBIS's unit tests and documentation tests, and must be PEP 8 compliant. We enforce these guidelines with our CI process. To run these tests locally, and for helpful tips on git, see our [Contribution Guide](.github/workflows/CONTRIBUTING.md).

IBIS's `develop` branch has the latest contributions. Pull requests should target `develop`, and users who want the latest package versions, features, etc. can use `develop`.


Contributions should be submitted as a pull request pointing to the `develop` branch, and must pass IBIS's CI process; to run the same checks locally, use:

```bash
pytest tests/test_*.py
```

## Releases
See our [change log](CHANGELOG.md) for more details.

## Code of Conduct
Please note that IBIS has a [Code of Conduct](.github/workflows/CODE_OF_CONDUCT.md). By participating in the IBIS community, you agree to abide by its rules.

## License
IBIS is distributed under the terms of the MIT license. All new contributions must be made under the MIT license. See LICENSE and NOTICE for details.

LLNL-CODE-838977