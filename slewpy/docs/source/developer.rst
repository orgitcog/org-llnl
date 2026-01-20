===============
Developer Guide
===============

In order to help develop popclass, the library should be installed from source.

.. code-block:: console

    git clone https://github.com/LLNL/slewpy.git

Working in the Dev Container
----------------------------

For convenience, we include a Dockerfile for development within the container.
Instructions on installing Docker and its basic usage can be found on the `Docker Website <https://www.docker.com/>`_.

To build the container:

.. code-block:: console

    cd popclass
    docker build ./ -t slewpy:latest

This will build the Docker image.
To check that it has built correctly, run the test suite:

.. code-block:: console

    docker run -it -v "./:/slewpy" slewpy:latest bash -c "cd slewpy/ && pytest -v"
