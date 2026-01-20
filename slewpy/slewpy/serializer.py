import csv

def targets_from_csv(path):
    """
    Read in targets from csv file and into Target objects.

    Args:
        path (str): File path to target list
    Returns:
        targets (List[Target]): List of targets 
    Raise:
        ValueError if csv column names don't match expected names.
    """
    expected_column_names = ['target_id','time_start', 'time_end', 'target_ra', 'target_dec', 'target_type']
    object_properties = ['id', 'time_start', ] 

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        if set(expected_column_names) != reader.fieldnames:
            raise ValueError(f"Expected column names
             {expected_column_names} got {reader.fieldnames}")


    return targets



