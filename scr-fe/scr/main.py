"""
SCR Frontend command line executable.
"""

import os
import sys
import click

class SCR(object):
    def __init__(self, home):
        self.home = sys.argv[0]
        self.config = {}
        self.verbose = False

    def set_config(self, key, value):
        self.config[key] = value
        if self.verbose:
            click.echo('  config[%s] = %s' % (key, value), file=sys.stderr)

    def get_config(self, key):
        if key in self.config:
            print(self.config[key])
        else:
            print("ERROR: %s not found" % key)

    def list(self):
        for key in self.config:
            print('%s = \t %s' % key, self.config[key])

    def __repr__(self):
        return "<SCR %r>" % self.home

pass_scr = click.make_pass_decorator(SCR)

@click.group()
@click.option('--config', nargs=2, multiple=True,
              metavar='KEY VALUE', help='Overrides a config key/value pair.')
@click.option('--verbose', '-v', is_flag=True,
              help='Enables verbose mode.')
@click.version_option('0.1')
@click.pass_context
def cli(ctx, config, verbose):
    """SCR is a command line tool for interacting with Scalable Checkpoint/Restart library."""
    # Create an SCR object and remember it as the context object
    # Other commands refer to it with the @pass_scr decorator
    ctx.obj = SCR(os.getcwd)
    ctx.obj.verbose = verbose
    for key, value in config:
        ctx.obj.set_config(key, value)

@cli.command()
@pass_scr
def test(scr):
    print(scr)

import sh

@cli.command()
@click.argument('hashfile')
@pass_scr
def print_hash(scr, hashfile):
    """Print contents of a hashfile to stdout."""
    kvtree_print_file = sh.Command('./kvtree_print_file')
    print(kvtree_print_file(hashfile))

@cli.command()
@pass_scr
def prefix(scr):
    """Print SCR Prefix directory"""
    if 'SCR_PREFIX' in os.environ:
        print(os.environ['SCR_PREFIX'])
    else:
        print(os.getcwd())

@cli.command()
@click.option('--user', '-u', is_flag=True,
              help='List the username of current job.')
@click.option('--jobid', '-j', is_flag=True,
              help='List the job id of the current job.')
@click.option('--nodes', '-n', is_flag=True,
              help='List the node-set the current job is using.')
@click.option('--down', '-d', is_flag=True,
              help='List any nodes of the job\'s allocation that resource manager knows to be down.')
@click.option('--runnodes', '-r', is_flag=True,
              help='List the number of nodes using the last run.')
@click.option('--endtime', '-e', is_flag=True,
              help='Lists the end time of the jobs.')
@click.option('--prefix', '-p',
              help='Specify the prefix directory.')
@pass_scr
def env(scr, user, jobid, nodes, down, runnodes, endtime, prefix):
    """Print details about the SCR enviroment"""
    pass

@cli.group()
@click.option('-n', metavar='N',
              help='limit to the N most recent filesets')
@pass_scr
def list(scr, n):
    """List fileset details."""
    pass

@cli.group()
@pass_scr
def setup(scr):
    """Setup the run script."""
    pass


@setup.command()
@click.argument('cmd', nargs=-1)
@pass_scr
def run(scr, cmd):
    """Set run command."""
    pass

@setup.command()
@click.argument('cmd', nargs=-1)
@pass_scr
def restart(scr, cmd):
    """Set restart command.
    If not set, defaults to run command."""
    pass


@cli.group()
@click.option('--globally', is_flag=True,
              help='Use global config file (~/.scrconfig)')
@click.option('--locally', is_flag=True,
              help='Use local config file (./.scr/config)')
@click.option('--file', '-f', metavar='<file>', type=click.File('rwb'),
              help='Use given config file')
@click.option('--name-only', is_flag=True,
              help='Show variable names only')
@click.option('--show-origin', is_flag=True,
              help='Show origin of config (file, standard input, blob, command line)')
@pass_scr
def config(scr, globally, locally, file, name_only, show_origin):
    """Sub-commands to configure SCR"""
    pass

@config.command()
@click.option('--all', is_flag=True,
              help='Get all values: key [value-regex]')
@click.option('--regex', is_flag=True,
              help='Get values for regexp: name-regex [value-regex]')
@click.argument('key')
@pass_scr
def get(scr, all, regex, key):
    """Get SCR config values."""
    scr.get_config(key)

@config.command()
@click.argument('key')
@click.argument('value')
@pass_scr
def add(scr, key, value):
    """Add a new variable: name [value-regex]"""
    scr.set_config(key, value)

@config.command()
@pass_scr
def list(scr):
    """List all"""
    scr.list()
    pass

@config.command()
@pass_scr
def edit(scr):
    """Open an editor"""
    pass

@config.command()
@click.option('--all', is_flag=True,
               help='Remove all matches: name [value-regex]')
@pass_scr
def unset(scr, all):
    """Remove a variable: name [value-regex]"""
    pass

# @click.option('--bool',
#               help='Value is "true" or "false"')
# @click.option('--int',
#               help='Value is decimal number')
# @click.option('--bool-or-int',
#               help='Value is --bool or --int')
# @click.option('--path',
#               help='Value is a path (file or directory name)')
