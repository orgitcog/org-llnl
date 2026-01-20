import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from scipy.signal import argrelextrema, savgol_filter
from typing import Optional
from orchestrator.utils.exceptions import CellTooSmallError
from orchestrator.utils.data_standard import METADATA_KEY


def extract_env(
    original_atoms: Atoms,
    rc: float,
    atom_inds: list[int],
    new_cell: np.ndarray,
    extract_cube: Optional[bool] = False,
    min_dist_delete: Optional[float] = 0.7,
    keys_to_transfer: Optional[list[str]] = None,
) -> list[Atoms]:
    """
    function for extracting local environments

    Requires ase and numpy. Written by Jared Stimac (documentation
    reformatted for Orchestrator), additional checks added.

    :param original_atoms: ase atoms object of the config you wish to extract
        an atoms local env
    :param rc: cuttoff radius to extract and constrain positions in Angstroms
    :param atom_ind: list of indices (0-based) for which atom you want to
        extract local environments.
    :param new_cell: ase Cell object (3x3 array) you wish to embed the
        environment into, expected to be cube
    :param extract_cube: specifies if you want to extract all atoms
        within a cube of the same size of new_cell.
        * NOTE: the when extracting a cube shape, only atoms within a sphere
        defined by rc will be constrained for potential relaxation (not
        currently performed)
    :param min_dist_delete: float dist in Angstroms specifies how close atoms
        need to be to one another, excluding those in the fixed center core,
        to be considered colliding and deleted. Set to 0 for no deletions.
        This is done to remove unphysically close contacts resulting from the
        new boundaries. |default| ``0.7``
    :param keys_to_transfer: list of array keys which contain additional data
        that should be attached to the new configurations
    :returns: list of ase atoms objects with the local environment emedded
    """
    if new_cell.size == 3:
        cell_norms = new_cell
        max_cell_len = np.max(new_cell)
    else:
        if ~(new_cell[np.where(~np.eye(new_cell.shape[0], dtype=bool))] == 0):
            raise ValueError('New cell is non orthorhombic; this is an '
                             'unxpected case not accounted for')

        cell_norms = np.linalg.norm(new_cell, 2, 1)
        max_cell_len = np.max(cell_norms)
    if keys_to_transfer is None:
        keys_to_transfer = []

    # initial checks
    if ~(cell_norms[0] == cell_norms[1] == cell_norms[2]):
        raise ValueError('New cell is not a cube. This is an unxpected case '
                         'not accounted for')
    if (max_cell_len > original_atoms.cell.cellpar()[0]
            or max_cell_len > original_atoms.cell.cellpar()[1]
            or max_cell_len > original_atoms.cell.cellpar()[2]):
        raise CellTooSmallError(
            'Requested extracted cell size is larger than original structure')
    if max_cell_len < rc:
        raise CellTooSmallError('The specified value for rc is greater than '
                                'the maximum cell vector length for the new '
                                'supercell')
    if rc * 2 > max_cell_len:
        raise CellTooSmallError('2*rc is greater than the extracted cell side '
                                'length')
    if isinstance(atom_inds, int):
        atom_inds = [atom_inds]

    # get neighboring atom pos displacements
    n_atoms = original_atoms.get_positions().shape[0]
    # cuttoff for each atom, uses overlapping spheres of rc, so only need 1/2
    # length, see docs
    cutoffs = (0.5 * max_cell_len * np.ones((n_atoms))).tolist()
    nl = NeighborList(cutoffs, self_interaction=True, bothways=True)
    nl.update(original_atoms)
    subcells = []
    key_arrays = {k: original_atoms.get_array(k) for k in keys_to_transfer}

    for atom_ind in atom_inds:
        indices, offsets = nl.get_neighbors(atom_ind)
        neigh_disp = np.zeros((offsets.shape[0], 3))
        neigh_symbols = []
        neigh_arrays = {k: [] for k in keys_to_transfer}
        neigh_ind = 0
        for i, offset in zip(indices, offsets):
            neigh_disp[neigh_ind, :] = (original_atoms.positions[i]
                                        + offset @ original_atoms.get_cell()
                                        ) - original_atoms.positions[atom_ind]
            neigh_symbols.append(original_atoms.symbols[i])
            for key in keys_to_transfer:
                neigh_arrays[key].append(key_arrays[key][i])
            neigh_ind += 1

        # find atoms in cube
        ind_in_cube = np.where(
            np.all(np.abs(neigh_disp) <= (max_cell_len / 2), 1))[0]
        cube_disp = neigh_disp[ind_in_cube, :]
        cube_symbols = []
        cube_arrays = {k: [] for k in keys_to_transfer}
        for i in ind_in_cube:
            cube_symbols.append(neigh_symbols[int(i)])
            for key in keys_to_transfer:
                cube_arrays[key].append(neigh_arrays[key][int(i)])

        # find atoms within rc
        ind_in_rc = np.where(np.linalg.norm(cube_disp, 2, 1) <= rc)[0]
        sphere_disp = cube_disp[ind_in_rc, :]
        sphere_symbols = []
        sphere_arrays = {k: [] for k in keys_to_transfer}
        for i in ind_in_rc:
            sphere_symbols.append(cube_symbols[int(i)])
            for key in keys_to_transfer:
                sphere_arrays[key].append(cube_arrays[key][int(i)])

        # make new atoms object
        if extract_cube:
            new_pos = cube_disp
            new_symbols = cube_symbols
            ind_fix = ind_in_rc
            new_arrays = cube_arrays
        else:
            new_pos = sphere_disp
            new_symbols = sphere_symbols
            ind_fix = np.arange(new_pos.shape[0], dtype=int)
            new_arrays = sphere_arrays

        box_center = cell_norms / 2
        new_pos = new_pos + box_center

        new_atoms = Atoms(symbols=new_symbols,
                          positions=new_pos,
                          cell=new_cell,
                          pbc=True)
        for key, arr in new_arrays.items():
            # data was saved as list of 1d arrays convert to 2D
            new_atoms.set_array(key, np.array(arr))
        # check info dict for any keys related to the keys_to_transfer
        new_info_dict = {}
        new_metadata_dict = {}
        for output_key in [x.rsplit('_', 1)[0] for x in keys_to_transfer]:
            for info_key in original_atoms.info:
                if output_key in info_key:
                    new_info_dict[info_key] = original_atoms.info[info_key]
            if output_key in original_atoms.info[METADATA_KEY]:
                new_metadata_dict[output_key] = original_atoms.info[
                    METADATA_KEY][output_key]
        new_atoms.info = new_info_dict
        new_atoms.info[METADATA_KEY] = new_metadata_dict
        # add constraint
        from ase.constraints import FixAtoms
        c = FixAtoms(indices=ind_fix)
        new_atoms.set_constraint(c)

        # delete atoms colliding at the new boundaries
        if extract_cube and min_dist_delete > 0:
            total_collisions, num_collisions_per_atom = _find_collisions(
                new_atoms, min_dist_delete)
            while total_collisions != 0:
                atom_to_delete = np.argmax(num_collisions_per_atom)
                del new_atoms[atom_to_delete]
                total_collisions, num_collisions_per_atom = _find_collisions(
                    new_atoms, min_dist_delete)

        subcells.append(new_atoms)

    return subcells


def find_central_atom(config: Atoms, side_size: float) -> int:
    """
    Find the central atom index in an extracted environment

    The extract_env function does not specify the index of the atom which was
    extracted. However, it is guaranteed to be in the center of the cell. This
    method uses this to find the index of the central atom.

    :param config: subcell within which the central atom will be found
    :param side_size: length of the cubic cell, half of which will be the
        central atom's coordinates in all three direction
    :returns: index of the central atom in the Atoms object
    """
    half_length = side_size / 2
    atom_found = False
    for i, atom_pos in enumerate(config.positions):
        for coord in atom_pos:
            if coord == half_length:
                atom_found = True
            else:
                atom_found = False
                break
        if atom_found:
            return i


def get_ith_shell(
    config: Atoms,
    central_atom_index: int,
    shell_index: int,
) -> np.ndarray:
    """
    Find the indices of atoms in the first neighbor shell of a central atom

    This function computes the radial distribution function (RDF) to estimate
    the first nearest neighbor (1NN) shell distance, then identifies all atoms
    within that shell around the specified central atom.

    :param config: Atoms object representing the atomic configuration
    :type config: Atoms
    :param central_atom_index: Index of the central atom whose neighbors are
        sought
    :type central_atom_index: int
    :param shell_index: what shell to extract, from 1 (1NN), up to N
        (within 10 A)
    :type shell_index: int
    :returns: Indices of atoms in the first shell as a numpy array
    :rtype: np.ndarray
    """
    if shell_index < 1:
        raise ValueError('shell_index must be at least 1')
    r, rdf = _get_rdf(config, 5.0, 0.1)
    # smooth the function for easier peak/valley extraction
    rdf_smooth = savgol_filter(rdf, window_length=10, polyorder=3)
    # get the peaks and valleys starting from global max
    peak_idxs, valley_idxs = _find_peaks_and_valleys(rdf_smooth)
    if shell_index > len(valley_idxs):
        raise RuntimeError(f'Requested {shell_index}NN shell but only '
                           f'{len(valley_idxs)} shells found within 10 A')
    shell_idx = valley_idxs[shell_index - 1]
    shell_distance = r[shell_idx]
    # now get ith NN shell of central atom
    cutoffs = [0.5 * shell_distance] * len(config)
    nl = NeighborList(cutoffs, skin=0.0, bothways=True, self_interaction=True)
    nl.update(config)
    indices, _ = nl.get_neighbors(central_atom_index)
    return indices


def _get_rdf(
    config: Atoms,
    r_max: float,
    dr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the radial distribution function for a given structure

    :param config: Atoms object representing the atomic configuration
    :type config: Atoms
    :param r_max: Maximum distance to consider for RDF calculation (Angstrom)
    :type r_max: float
    :param dr: Bin width for RDF histogram (Angstrom)
    :type dr: float
    :returns: bin centers (r) and RDF values (rdf)
    :rtype: Tuple(np.ndarray)
    """
    n_atoms = len(config)
    cell = config.get_cell()
    positions = config.get_positions()
    cutoffs = [0.5 * r_max] * n_atoms
    nl = NeighborList(cutoffs, skin=0.0, bothways=True, self_interaction=False)
    nl.update(config)
    bins = np.arange(0, r_max + dr, dr)
    rdf_hist = np.zeros(len(bins) - 1)
    for i in range(n_atoms):
        indices, offsets = nl.get_neighbors(i)
        pos_i = positions[i]
        for j, offset in zip(indices, offsets):
            if i == j:
                # this should not happen with self_interaction = False
                continue
            pos_j = positions[j] + np.dot(offset, cell)
            dist = np.linalg.norm(pos_j - pos_i)
            if dist < r_max:
                bin_idx = int(dist // dr)
                if bin_idx < len(rdf_hist):
                    rdf_hist[bin_idx] += 1
    # Normalize
    r = 0.5 * (bins[:-1] + bins[1:])
    shell_volumes = 4.0 / 3.0 * np.pi * (bins[1:]**3 - bins[:-1]**3)
    number_density = n_atoms / config.get_volume()
    rdf = rdf_hist / (n_atoms * shell_volumes * number_density)
    return r, rdf


def _find_peaks_and_valleys(rdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the indicies of the peaks and valleys in the RDF

    Locates the first major peak in the RDF and then finds the position of the
    following valleys and peaks, which are used to define the boundary of the
    neighbor shells.

    :param rdf: RDF values
    :type rdf: np.ndarray
    :returns: indices of the peaks and valleys starting from the global max
    :rtype: Tuple(float)
    """
    # Find index of the largest peak
    max_peak_idx = np.argmax(rdf)
    # Find all the peaks and valleys
    peak_indices = argrelextrema(rdf, np.greater)[0]
    valley_indices = argrelextrema(rdf, np.less)[0]
    # Filter indices after the max peak
    peak_indices_from_max = peak_indices[peak_indices >= max_peak_idx]
    valley_indices_from_max = valley_indices[valley_indices > max_peak_idx]

    return peak_indices_from_max, valley_indices_from_max


def _find_collisions(atoms: Atoms, min_dist: float) -> tuple[int, np.ndarray]:
    """
    Find the collisions between atoms that do not have constraints.

    Used in extract_env() to find collisions across boundaries so those atoms
    can be deleted

    :param atoms: atoms for which to search for collisions
    :type: ASE Atoms object
    :param min_dist: float criterion below or equal to which is considered
        a collision between atoms
    :type min_dist: float
    :returns: (total_collisions) total number of unique collisions followed
        by (num_collisions_per_atom) array containing number of collisions
        per atom
    :rtype: tuple[int, np.ndarray]
    """
    # get neighbors that constitute collisions
    n_atoms = len(atoms)
    cutoffs = (0.5 * min_dist * np.ones((n_atoms))).tolist()
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0)
    nl.update(atoms)
    # find atoms with most collisions
    connect_mat = nl.get_connectivity_matrix(sparse=False)
    num_collisions_per_atom = np.sum(connect_mat, 1)
    # ignore atoms in the fixed core
    for const in atoms.constraints:
        num_collisions_per_atom[const.get_indices()] = 0
    total_collisions = int(np.sum(num_collisions_per_atom) / 2)
    return (total_collisions, num_collisions_per_atom)
