<?php

namespace LLNL\Phpmd\Blt\Plugin\Commands;

use Acquia\Blt\Robo\BltTasks;
use Acquia\Blt\Robo\Common\YamlMunge;
use Acquia\Blt\Robo\Exceptions\BltException;

/**
 * Defines commands in the "test" namespace.
 */
class PHPMessDetectorCommand extends BltTasks {

  /**
   * PHP Mess file suffixes.
   *
   * @var string
   */
  protected $fileSuffixes;

  /**
   * PHP Mess bin path.
   *
   * @var string
   */
  protected $phpmdBin;

  /**
   * PHP Mess report format.
   *
   * @var string
   */
  protected $reportFormat;

  /**
   * PHP Mess rules.
   *
   * @var string
   */
  protected $rules;

  /**
   * PhpMessCommands constructor.
   */
  public function __construct() {
    $this->phpmdBin = 'vendor/bin/phpmd';
    $this->rules = 'phpmd';
    $this->reportFormat = 'text';
    $this->fileSuffixes = '--suffixes php,module,inc,theme,profile';
  }

  /**
   * Run the PHPMD on the custom modules, themes and profiles.
   *
   * @command validate:phpmd:files
   * @aliases phpmda validate:phpmd:files validate:phpmd
   * @description Run the PHPMD (PHP Mess Detector) on the custom modules and custom themes.
   */
  public function scanAll() {
    $this->say("Running PHP Mess on custom modules and themes...");
    $custom_module_path = 'docroot/modules/custom';
    $custom_theme_path = 'docroot/themes/custom';
    $custom_profile_path = 'docroot/profiles/custom';

    $exclude_module_paths = [];
    $exclude_module = '';
    $exclude_modules = $this->getConfigValue('phpmd.exclude') ?: [];
    foreach ($exclude_modules as $module) {
      $exclude_module_paths[] = "$custom_module_path/$module";
    }

    if (!empty($exclude_module_paths)) {
      $exclude_module = "--exclude " . implode(',', $exclude_module_paths);
    }

    $is_module_success = $this->taskExec("$this->phpmdBin $custom_module_path $exclude_module $this->reportFormat $this->rules $this->fileSuffixes")->run()->wasSuccessful();
    $is_theme_success = $this->taskExec("$this->phpmdBin $custom_theme_path $this->reportFormat $this->rules $this->fileSuffixes")->run()->wasSuccessful();
    $is_profile_success = $this->taskExec("$this->phpmdBin $custom_profile_path $this->reportFormat $this->rules $this->fileSuffixes")->run()->wasSuccessful();

    $has_failed_phpmd_validation = !$is_module_success || !$is_theme_success || !$is_profile_success;

    if ($has_failed_phpmd_validation) {
      throw new BltException("PHPMD Scan failed.");
    }
  }

  /**
   * Run PHPMD on a given path.
   *
   * This command will run PHPMD against a list of files provided in a comma
   * separated list.
   *
   * @command validate:phpmd:file
   * @options pre-commit Whether or not to consider during pre-commit.
   * @aliases phpmdf validate:phpmd:file
   * @description Run the PHPMD on given path.
   */
  public function scanPath($path, $options = ['pre-commit' => 'no']) {
    if (empty($path)) {
      return;
    }

    if ($options['pre-commit'] === 'yes') {
      $path = preg_replace("/\r|\n/", ",", $path);
    }
    $this->say("Running PHPMD on $path...");
    $is_success = $this->taskExec("$this->phpmdBin $path $this->reportFormat $this->rules $this->fileSuffixes")->run()->wasSuccessful();

    if (!$is_success) {
      throw new BltException("PHPMD failed on $path.");
    }
  }

  /**
   * Initializes default template toggle modules for this project.
   *
   * @command recipes:config:init:phpmd
   *
   * @throws \Acquia\Blt\Robo\Exceptions\BltException
   */
  public function generateToggleModulesConfig() {
    $this->say("This command will automatically generate template phpmd exclude settings for this project.");
    // Sets default values for the project's blt.yml file.
    $project_yml = $this->getConfigValue('blt.config-files.project');
    $this->say("Updating {$project_yml}...");
    $project_config = YamlMunge::parseFile($project_yml);
    $project_config['phpmd']['exclude'] = [];
    try {
      YamlMunge::writeFile($project_yml, $project_config);
      $this->say("Please edit your project blt.yml file with desired phpmd exclude settings.");
      $this->say("It is also valid to leave this setting blank.");
    }
    catch (\Exception $e) {
      throw new BltException("Unable to update $project_yml.");
    }
  }
}
