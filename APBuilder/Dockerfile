ARG BASE_REGISTRY=ghcr.io
ARG BASE_IMAGE=llnl/apbuilder/ironbank/redhat/ubi8
ARG BASE_TAG=8.10
ARG PYTHON_VERSION=3.12
ARG MICROMAMBA_VERSION=2.3.3-0

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG} AS builder
# re-declare and auto inherit the value set earlier for PYTHON_VERSION
ARG PYTHON_VERSION
WORKDIR /build

COPY pyproject.toml .
COPY src ./src
COPY .git ./.git
COPY dist ./dist

RUN if [ -z "$(ls dist/*.whl 2>/dev/null)" ]; then \
    yum -y install git python${PYTHON_VERSION} python${PYTHON_VERSION}-pip && \
    python3 -m pip install --upgrade pip setuptools && \
    python3 -m pip install build wheel setuptools-scm && \
    python3 -m build --no-isolation . -o dist/; \
    fi

FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG}
ARG USER=apbuilder
# re-declare and auto inherit the value set earlier for PYTHON_VERSION
ARG PYTHON_VERSION
ARG MICROMAMBA_VERSION

# Create user to avoid using root
RUN groupadd -g 1002 ${USER} && \
    useradd -r -m -u 1002 -g ${USER} ${USER}

# Install and configure micromamba
RUN curl -Ls https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64 \
    -o /usr/local/bin/micromamba && \
    chmod +x /usr/local/bin/micromamba && \
    micromamba --version

USER ${USER}
ENV VENV_NAME=apbuilder-venv

# Create environment with dependencies
RUN echo ${VENV_NAME}
RUN micromamba create -y -n ${VENV_NAME} python=${PYTHON_VERSION} \ 
    'pygmt=0.17' \
    'libgdal-grib' \
    'libgdal-netcdf' \
    -c conda-forge && \
    micromamba clean -yaf

# Make RUN commands using the new environment:
# Unfortunately, SHELL command does not support use of ARG variables, 
# hence the name of the virtual environment is hardcoded here. 
# It must be the same value as argument VENV_NAME defined at the top.
SHELL ["micromamba", "run", "-n", "apbuilder-venv", "/bin/bash", "-c"]

COPY --from=builder --chown=${USER}:${USER} /build/dist/ /home/${USER}/dist

RUN python3 -m pip install /home/${USER}/dist/*.whl && \
    rm -rf /home/${USER}/dist

WORKDIR /home/${USER}
COPY --chown=${USER} src/apbuilder/config/config.toml .config/herbie/config.toml
RUN mkdir -p logs && touch logs/.ignore

# Unfortunately, ENTRYPOINT command does not support use of ARG or ENV variables directly,
# hence the name of the virtual environment is hardcoded here. 
# It must be the same value as argumenta VENV_NAME defined at the top of this file.
ENTRYPOINT ["micromamba", "run", "-n", "apbuilder-venv"]
CMD ["apbuilder", "-h"]

ARG VERSION
ARG BUILD_DATE
ARG COMMIT_HASH
ARG NAME="APBuilder"
ARG DESCRIPTION="Atmospheric Profile Builder (APBuilder)"
ARG LICENSE="BSD-Commercial"
ARG VENDOR="LLNL"
ARG AUTHOR="Raul Viera-Mercado (vieramercado1@llnl.gov)"
LABEL name=$NAME
LABEL description=$DESCRIPTION
LABEL distribution-scope=$LICENSE
LABEL maintainer=$AUTHOR
LABEL build-date=$BUILD_DATE
LABEL release=""
LABEL summary=$DESCRIPTION
LABEL url=""
LABEL vcs-ref=""
LABEL vcs-type=""
LABEL vendor=$VENDOR
LABEL version=$VERSION
LABEL git_commit=$COMMIT_HASH
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.name=$NAME
LABEL org.opencontainers.image.title="Atmospheric Profile Builder"
LABEL org.opencontainers.image.description=$DESCRIPTION
LABEL org.opencontainers.image.vendor=$VENDOR
LABEL org.opencontainers.image.authors=$AUTHOR
LABEL org.opencontainers.image.licenses=$LICENSE
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.revision=""
LABEL org.opencontainers.image.base.name="ironbank/redhat/ubi/ubi8:8.10"
LABEL gov.llnl.image.ironbank="True"
LABEL gov.llnl.tags="llnl, gmp, apbuilder, geophysics, python"
