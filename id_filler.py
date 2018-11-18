import sys

test = sys.argv[1]

target = sys.argv[2]

have_ids = set()
with open(target) as infile:
    for line in infile:
        have_ids.add(line.split(",")[0].strip())
not_have_ids = []
with open(test) as infile:
    for line in infile:
        id = line.split(",")[0].strip().strip("\"")
        if id not in have_ids:
            not_have_ids.append(id)
with open(target, "a") as outfile:
    for id in not_have_ids:
        outfile.write(id + ", \n")
