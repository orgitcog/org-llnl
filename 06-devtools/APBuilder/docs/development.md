# Development of APBuilder

Install all pre-requisites as defined in the [home page](install.md#pre-requisites).

## Software Development

Using `git`, clone the repo.

Install the Python application locally, including all development and testing dependencies,
using the following command:

````bash
pip install -e .[dev,test]
````

### Running unit test

The unit tests are using the `pytest` framework.
Run the following command in your terminal to execute unit test:

```bash
pytest
```

This will generate a code coverage report accessible in the `coverage-reports` directory.
It includes an HTML report accessible in `htmlcov\index.html`, which you can open in your preferred web browser.

## Documentation Development

The documentation is developed using MkDocs and files are located in the `docs` directory.  
To develop the documentation you must install the following:

````bash
pip install -e .[docs]
````

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it.
Make sure you're in the same directory as the `mkdocs.yml` configuration file, and then start the
server by running the mkdocs serve command.

````bash
mkdocs serve
````

## Release Process

Here are the steps to make a new release with a version number using semantic versioning.

1. Create a new branch from the `main` branch
1. On the new branch, update the `CHANGELOG.md`
    1. Replace `\[Unreleased\]` with the version number
    1. Replace `yyyy-mm-dd` with the current date
1. Commit changes
1. Create MR and merge into `main` branch
1. On GitLab, create `New Release`
    1. Tag Name: version number
    1. Release Title: version number
    1. Everything else as default

The CI pipeline will be triggered by the tag creation.
This will build and deploy the Python package to the GitLab Package Registry.  
Once the pipeline is completed, follow this steps:

1. Go to the `Package Registry` page
1. Click on the new version
1. Copy the URL
1. Go to the `Releases` page
1. Click on the new version
1. Click on `Edit release`
1. Add a new `Release assets`
    1. URL: the URL of the package
    1. Link title: `Python Package`
    1. Type: `Package`

This completes the release process of a new version.  
Then you should update the `CHANGELOG.md` on the `main` branch to have it ready for new development.

### Default Changelog

```md
## \[Unreleased\] - yyyy-mm-dd

### Added

### Changed

### Deprecated

### Fixed

### Removed

### Security
```
