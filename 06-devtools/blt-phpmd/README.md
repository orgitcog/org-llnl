Acquia BLT PHPMD testing integration
====

This is an [Acquia BLT](https://github.com/acquia/blt) plugin providing a process for testing code using [PHP Mess Detector](https://phpmd.org).

LLNL does not provide any direct support for this 
software or provide any warranty as to its stability.

## Installation and usage

To use this plugin, you must already have a Drupal project using BLT 13.

1. Add this plugin to your project using composer:

`composer require llnl/blt-phpmd`

2. Initialize the exclude settings for your project:

`blt recipes:config:init:phpmd`

3. Update your `blt.yml` file with the list of modules you wish to exclude from phpmd validation. It is
valid to have no modules to exclude from testing.

4. Copy the `phpmd.xml.dist` file to `phpmd.xml` in the root directory of your project and configure rules according to [PHPMD](https://phpmd.org/documentation/)

There are two commands included `validate:phpmd:files` and `validate:phpmd:file`. 

`validate:phpmd:files` will check all files in the `docroot/modules/custom`, `docroot/themes/custom`, and
`docroot/profiles/custom` paths.

`validate:phpmd:file` expects a comma separated list of files to test. If given a directory, it will test
all files in that path. This is useful for CI testing of files changed or in local testing while coding.

# Release

See [LICENSE](LICENSE) and [NOTICE](NOTICE).

SPDX-License-Identifier: GPL-2.0-or-later

LLNL-CODE-839162
