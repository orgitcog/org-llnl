from os import system
import numpy as np
from orchestrator.utils.input_output import ase_glob_read
from orchestrator.storage import storage_builder
from orchestrator.utils.data_standard import (
    ENERGY_KEY,
    FORCES_KEY,
    METADATA_PROPERTY_MAP,
    SELECTION_MASK_KEY,
    SELECTOR_PROPERTY_MAP,
    SELECTOR_PROPERTY_DEFINITION,
)

storage = storage_builder.build(
    "COLABFIT", {
        'credential_file':
        '/PATH/TO/'
        'unittests_colabfit_credentials.json'
    })

# check if tables exist, if not set the defaults up
existing_properties = [
    _['property-name'] for _ in storage.get_property_definitions()
]
if 'energy' not in existing_properties:
    storage.setup_tables()

parameters = {
    # Fill both values with the relevant input parameters from the simulation.
    # This example is for Quantum Espresso.
    'code': {
        'SYSTEM': {
            'ecutwfc': 60  # Ry
        }
    },
    # The DFT oracle used for the simulations should have a
    # translate_universal_parameters() function that can be called and passed
    # the values from the `code` section.
    'universal': {
        'code': 'Quantum Espresso',
        'version': 'v7.4.1',
        'planewave_cutoff': 816  # eV
    }
}

metadata = {
    'description': 'Description of the data.',
    'authors': 'author list',
    'parameters': parameters
}

# see if Si dataset needs to be added
if storage.check_if_dataset_name_unique('unit_test_sample_configs'):
    configs = ase_glob_read('sample_configs')
    # make a custom property map because we don't have stress in this dataset
    storage.set_property_map({
        'energy_field':
        ENERGY_KEY,
        'force_field':
        FORCES_KEY,
        METADATA_PROPERTY_MAP['new_property_name']:
        METADATA_PROPERTY_MAP['new_map'],
    })
    handle = storage.new_dataset('unit_test_sample_configs', configs, metadata)
    print(f'Added sample configs as {handle} - remember to update the input '
          'files with the new handle!')

# see if Ta dataset needs to be added
if storage.check_if_dataset_name_unique('Ta_training_unit_test_configs'):
    configs = ase_glob_read('Ta_training_configs')
    # can use default map (energy, forces, stress, metadata) here
    storage.set_default_property_map()
    # for weighted training, apply arbitrary but deterministic mask
    for config in configs:
        config.set_array(SELECTION_MASK_KEY, (np.arange(len(config)) % 2 == 0))
    # add to property map
    storage.add_property_mapping(
        SELECTOR_PROPERTY_MAP['new_property_name'],
        SELECTOR_PROPERTY_MAP['new_map'],
    )
    # make sure the property defintion exists
    if SELECTOR_PROPERTY_DEFINITION[
            'property-name'] not in existing_properties:
        storage.define_new_properties(SELECTOR_PROPERTY_DEFINITION)

    handle = storage.new_dataset(
        'Ta_training_unit_test_configs',
        configs,
        metadata,
    )
    print(f'Added Ta configs as {handle} - remember to update the input '
          'files with the new handle!')

system('rm orch.log')
