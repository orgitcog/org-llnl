VARIORUM MIRRORS
================

This repository contains the spack mirrors of variorum's dependencies. These
mirrors are used in variorum's CI on GitHub Actions, specifically to install
the hwloc and jansson dependencies.

To create a new dependency:

    $ spack mirror create -d <path-to-repo> <dependency-name>

The above should create a new folder with the <dependency-name>, and another
folder in `_source-cache/archive` with a unique two-character ID. The former
should contain a tar file with a soft link to a tar file in the
`_source-cache/archive/<ID>` folder.

Here's an example:

    $ echo ${PWD}
    $HOME/variorum-spack-mirrors

    $ spack mirror create -d ${PWD} jansson

The above command creates a new folder `f2` in `_source-cache/archive` with the
following tar file:

    $ ls _source-cache/archive/f2/
    f22901582138e3203959c9257cf83eba9929ac41d7be4a42557213a22ebcc7a0.tar.gz

It also creates a `jansson` folder.

    $ ls -l jansson/
    lrwxrwxrwx 1 user user 99 Jul 29 17:19 jansson-2.13.1.tar.gz -> ../_source-cache/archive/f2/f22901582138e3203959c9257cf83eba9929ac41d7be4a42557213a22ebcc7a0.tar.gz

Commit these two new folders to the repository.
