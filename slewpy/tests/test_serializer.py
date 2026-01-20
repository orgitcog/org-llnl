import csv
from slewpy.serializer import targets_from_csv


def test_csv_serializer():

    with open('tests/test.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        column_names = reader.fieldnames
        print(column_names)
        for row in reader:
            print(row)
        
        d
