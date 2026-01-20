## Productivity Frameworks for HPC

Scientific discovery is increasingly enabled by productivity
frameworks not designed for high-performance computing
(HPC). High-level languages such as Python, machine learning
frameworks, and container technologies allow
users to rapidly develop, test, and execute their science algorithms
in a variety of platforms. While great headway has been made to run
these technologies on HPC systems, their portability and performance
can be limited in that the communication libraries they use may not be
a good match for what performs best on the HPC system. 

Productivity frameworks for HPC includes container recipes, build
recipes, continuous integration scripts, and other software aimed at
testing the portability of containerized HPC software across platforms
and interconnects. In particular, it tests the utility of
bind-mounting at the MPI layer (rather than the underlying fabric
layer) to leverage a standardized protocol and avoid various technical
debt and vendor lock-in. Since MPIs are often ABI-incompatible,
trampolines such as the open-source Wi4MPI library is used when such
cases arise.

<!---
### Getting started
--->

### Contributing 

Contributions for bug fixes and new features are welcome and follow
the GitHub
[fork and pull model](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-collaborative-development-models).
Contributors develop on a branch of their personal fork and create
pull requests to merge their changes into the main repository. 


1. [Fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) `productivity-frameworks`.
2. [Clone](https://help.github.com/en/github/getting-started-with-github/fork-a-repo#keep-your-fork-synced)
your fork: `git clone git@github.com:[username]/productivity-frameworks.git`
3. Create a topic branch for your changes: `git checkout -b new_feature`
4. Create feature or add fix (and add tests if possible)
5. Make sure everything still passes: `make check`
6. Push the branch to your GitHub repo: `git push origin new_feature`
7. Create a pull request against `productivity-frameworks` and describe what your changes do and why you think it should be merged. List any
outstanding *todo* items. 


### Authors

Edgar A. Leon, Nathan Hanford, and Eric Green.


### License

Productivity frameworks for HPC is distributed under the terms of the
MIT license. All new contributions must be made under this license. 

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

SPDX-License-Identifier: MIT.

LLNL-CODE-846462.
