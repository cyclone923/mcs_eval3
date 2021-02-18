import csv

with open("Gravity_test_output") as f:
    reader = csv.reader(f, delimiter="\,")
    for i in reader:
        print("printing!!!",i[0])