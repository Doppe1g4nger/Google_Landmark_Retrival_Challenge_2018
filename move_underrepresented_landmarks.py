import argparse
import os
import distutils.dir_util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move folders with less than N files in them to another directory")
    parser.add_argument(
        "-i", "--in_dir",
        type=str,
        help="Directory to move files from"
    )
    parser.add_argument(
        "-n", "--threshold",
        type=int,
        help="Lower limit needed for folder to remain untouched"
    )
    parser.add_argument(
        "-o", "--out_dir",
        type=str,
        help="Directory to move files to"
    )
    args = parser.parse_args()
    counter = 0
    print("Total number of classes in directory: {}".format(len(os.listdir(args.in_dir))))
    for directory in os.listdir(args.in_dir):
        full_path = os.path.join(args.in_dir, directory)
        if len(os.listdir(full_path)) < args.threshold:
            counter += 1
            util.copy_tree(full_path, os.path.join(args.out_dir, directory))
            util.remove_tree(full_path)
    print("There were {} landmarks with too few examples".format(counter))
