# Installing APBuilder
<!-- markdownlint-disable MD024 -->

If you are looking to upgrade apbuilder, you can skip to the [Upgrade](#upgrade) section.

## Types of Installation

There are 2 ways to install the application.

1. [Native install](#native-install) using mamba
1. [Container install](#container-install)

## Native Install

### Pre-requisites

Create mamba virtual environment

````bash
mamba create -y --name apbuilder python=3.11
mamba activate apbuilder
````

Before installing the tool, `PyGMT` must be install in the system.

````bash
mamba install -y -c conda-forge 'pygmt=0.17' libgdal-grib libgdal-netcdf
````

### Installation

Install the Python application, including all dependencies, using the following command:

````bash
pip install apbuilder
````

### Verify

You can run a selfcheck to make sure there are no errors with 3rd party dependencies.

```bash
apbuilder info --selfcheck
```

## Container Install

APBuilder can be run as a container. We provide a docker compose file for ease of use.  
You can follow the steps below to configure and run the container.

### Pre-requisites

To install the container image, you will need a container runtime, for example Docker or Podman.  
This instructions are written using docker.

### Download

Download the docker compose and volume override files.

1. Docker Compose
1. Volume Override

### Configure

In the `volume-overrides.yaml` change the following:

1. `<data-source>` to the full root path of where you want to store the data. A `data` and `output` directory will be automatically created in this root path.

### Verify

Run the following command to verify installation and configuration.  
It should print the apbuilder help message.  
NOTE: The first time it will take a little while because it will download the container image.

```bash
docker compose \
  -f apbuilder-docker-compose.yaml \
  -f volume-overrides.yaml run \
  --rm apbuilder
```

### Usage

You can run with docker compose using the following command:

```bash
docker compose \
  -f apbuilder-docker-compose.yaml \
  -f volume-overrides.yaml run \
  --rm apbuilder \
  apbuilder info --selfcheck
```

The actual command we are telling the container to execute is `apbuilder info --selfcheck`,
which you can change to execute any apbuilder command.

For example, try replacing it with `info -h`

The following prefix is always needed to run apbuilder with docker compose.

```bash
docker compose \
  -f apbuilder-docker-compose.yaml \
  -f volume-overrides.yaml \
  run --rm apbuilder
```

It is recommended to to create an alias to simplify usage:

```bash
alias apbuilder='
  docker compose \
  -f apbuilder-docker-compose.yaml \
  -f volume-overrides.yaml \
  run --rm apbuilder apbuilder'
```

NOTE: The word apbuilder is mentioned twice at the end of the command. This is not a mistake.  
The first instance is the name of the container. The seconds instance is the command we are executing inside the container.

Then you can easily run with:

```bash
apbuilder info -h
```

To learn more on how to use apbuilder, please visit the [Using APBuilder page](usage.md).

## Upgrade

To upgrade APBuilder along with herbie-data, use the following command.
Note this will not update transitive dependencies. It will only update the explicit dependencies in APBuilder.

```bash
pip install apbuilder --upgrade
```
