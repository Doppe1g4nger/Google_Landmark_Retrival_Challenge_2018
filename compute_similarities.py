# Modified from label_image.py from the tensorflow examples GitHub repository

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random
import time
import argparse
import pickle
import os.path
import multiprocessing

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KDTree


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_graph(model_f):
    # Copied from label_image, loads tensorflow graph
    g = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_f, "rb") as f:
        graph_def.ParseFromString(f.read())
    with g.as_default():
        tf.import_graph_def(graph_def)
    return g


def read_tensor_from_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_decoded, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resize = tf.image.resize_bilinear(dims_expander, [299, 299])
    normalized = tf.divide(tf.subtract(resize, [0]), [255])
    return normalized


if __name__ == "__main__":
    # Parse user inputs, needs test and index sets as well as where to save submission.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        help="graph/model to be executed"
    )
    parser.add_argument(
        "--test_set",
        help="directory of images to test on",
        required=True
    )
    parser.add_argument(
        "--index_set",
        help="directory of images to pull from",
        required=True
    )
    parser.add_argument(
        "--saved_features",
        help="Pickle file of dictionary mapping of image labels to their feature vectors"
    )
    parser.add_argument(
        "--submission_file",
        help="Path to file to save results to",
        required=True
    )
    parser.add_argument(
        "--cluster_subset_size",
        help="Number of samples to take when performing clustering for index set search space reduction",
        default=75000,
        type=int
    )
    parser.add_argument(
        "--num_clusters",
        help="Number of clusters to partition index set into",
        default=1600,
        type=int
    )
    parser.add_argument(
        "--kd_tree",
        help="Flag to use kd_tree rather than clustering",
        action="store_true"
    )
    args = parser.parse_args()

    test_ids = []
    test_vectors = []
    index_ids = []
    index_vectors = []

    index_files = [os.path.join(args.index_set, file) for file in os.listdir(args.index_set)]
    test_files = [os.path.join(args.test_set, file) for file in os.listdir(args.test_set)]
    all_files = set(index_files + test_files)

    # Load in pre-generated features if they already exist and are provided, otherwise generate new ones
    if args.saved_features:
        print("Loading in features...")
        test_ids, test_vectors, index_ids, index_vectors = pickle.load(open(args.saved_features, "rb"))
        print(len(test_ids), len(test_vectors), len(index_ids), len(index_vectors))
        print(len(all_files))

    if len(test_ids) + len(index_ids) < len(all_files):
        print("Starting feature generation...")

        test_ids_set = set(test_ids)
        index_ids_set = set(index_ids)

        existing_test_paths = [file for file in test_files if file.split("/")[-1].split(".")[0] in test_ids_set]
        existing_index_paths = [file for file in index_files if file.split("/")[-1].split(".")[0] in index_ids_set]
        existing_paths = set(existing_test_paths + existing_index_paths)

        to_process = list(all_files - existing_paths)

        feature_path = input("Please input a path to save the newly created features from:")
        will_overwrite_if_exists = False
        while os.path.exists(feature_path) and not will_overwrite_if_exists:
            path_overwrite = input("Entered path already exists, do you want to overwrite it? (y/n): ")
            if path_overwrite.upper() == "Y":
                will_overwrite_if_exists = True
            else:
                feature_path = input("Please input a path to save the newly created features from:")

        # Set default graph path, specify new one if provided, and load graph
        model_file = "/u/spa-d2/ugrad/dbdo224/Data/Landmark_Retrieval/" \
                     "TensorFlow_Playground/inception_v3_2016_08_28_frozen.pb"
        if args.graph:
            model_file = args.graph
        graph = load_graph(model_file)
        input_operation = graph.get_operation_by_name("import/input")
        output_tensor = graph.get_tensor_by_name("import/InceptionV3/Logits/Dropout_1b/Identity:0")

        with tf.Session(graph=graph) as sess:
            x = tf.placeholder(tf.float32)
            normed = tf.norm(x)

            data_set = tf.data.Dataset.from_tensor_slices(tf.constant(to_process))
            data_set = data_set.map(read_tensor_from_image)
            next_el = data_set.make_one_shot_iterator().get_next()

            test_files_set = set(test_files)

            for file in tqdm(to_process, desc="Progress of image feature generation"):
                image_tensor = sess.run(next_el)
                results = sess.run(output_tensor, {
                    input_operation.outputs[0]: image_tensor
                })
                results = np.squeeze(results)
                results = (results / sess.run(normed, {x: results})).reshape(1, -1)
                if file not in test_files_set:
                    index_ids.append(file.split("/")[-1].split(".")[0])
                    index_vectors.append(results)
                else:
                    test_ids.append(file.split("/")[-1].split(".")[0])
                    test_vectors.append(results)

            # Write generated features to file, TODO Add error handling in case file save fails
            pickle.dump((test_ids, test_vectors, index_ids, index_vectors), open(feature_path, "wb"))

    if not os.path.exists("cluster_vectors.p") and not args.kd_tree:
        print("Beginning clustering...")
        random.seed(460)
        start = time.time()
        sub_matrix = np.vstack(random.choices(index_vectors, k=args.cluster_subset_size))
        cluster_generator = AgglomerativeClustering(n_clusters=args.num_clusters)
        cluster_assignments = cluster_generator.fit_predict(sub_matrix)

        clusters = [[] for i in range(args.num_clusters)]
        for i in range(len(cluster_assignments)):
            clusters[cluster_assignments[i]].append(sub_matrix[i])
        representative_vectors = np.vstack([np.mean(np.vstack(cluster), axis=0) for cluster in clusters])

        print("Num seconds taken to cluster:", time.time() - start)
        pickle.dump(representative_vectors, open("cluster_vectors.p", "wb"))
    elif os.path.exists("cluster_vectors.p") and not args.kd_tree:
        print("Loading representative vectors...")
        representative_vectors = pickle.load(open("cluster_vectors.p", "rb"))
    if not args.kd_tree:
        clusters = [[] for i in range(args.num_clusters)]
        for i in tqdm(range(len(index_vectors)), desc="Assigning index vectors to clusters"):
            highest_similarity = np.dot(index_vectors[i], representative_vectors[0])
            cluster_index = 0
            for j in range(1, len(representative_vectors)):
                similarity = np.dot(index_vectors[i], representative_vectors[j])
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    cluster_index = j
            clusters[cluster_index].append((index_ids[i], index_vectors[i]))
        clusters = [cluster for cluster in clusters if len(cluster) != 0]
        cluster_lengths = [len(cluster) for cluster in clusters]
        representative_vectors = []
        for cluster in clusters:
            representative_vectors.append(np.mean(np.vstack([item[1] for item in cluster]), axis=0))
        print("Total clusters: {}".format(len(clusters)))
        print("Largest cluster has {} elements".format(max(cluster_lengths)))
        print("Smallest cluster has {} elements".format(min(cluster_lengths)))
        print("Median cluster size: {}".format(np.median(cluster_lengths)))

        # Compute similarities and write results to file
        representative_matrix = np.vstack(representative_vectors)
        with open(args.submission_file, "w") as submission:
            # First line of submission based on Landmark Retrieval challenge formatting guidelines
            submission.write("id,images\n")
            with tf.Session() as sess:
                x = tf.placeholder(tf.float32)
                y = tf.placeholder(tf.float32)
                multiply = tf.matmul(x, y, transpose_b=True)
                # For each test image generate a list of tuples containing similarity, index id pairs
                for i in tqdm(range(len(test_vectors)), desc="Computing similarities"):
                    closest_cluster_indices = []
                    closest_clusters = []
                    total_examples = 0
                    cluster_similarities = np.squeeze(sess.run(multiply, {x: test_vectors[i], y: representative_matrix}))
                    while total_examples < 100:
                        highest_similarity = float("-inf")
                        cluster_index = -1
                        for j in range(len(cluster_similarities)):
                            if cluster_similarities[j] > highest_similarity and j not in closest_clusters:
                                cluster_index = j
                                highest_similarity = cluster_similarities[j]
                        total_examples += len(clusters[cluster_index])
                        closest_cluster_indices.append(cluster_index)
                        index_subset_ids = [item[0] for item in clusters[cluster_index]]
                        index_subset_matrix = np.vstack([item[1] for item in clusters[cluster_index]])
                        closest_clusters.append((index_subset_ids, index_subset_matrix))
                    image_comparisons = []
                    for index_subset_ids, index_subset_matrix in closest_clusters:
                        similarities = np.squeeze(sess.run(multiply, {x: test_vectors[i], y: index_subset_matrix})).reshape((1,-1))
                        image_comparisons += [(similarities[0][i], index_subset_ids[i]) for i in range(len(similarities[0]))]
                    image_comparisons.sort()
                    submission.write(test_ids[i] + "," + " ".join([item[1] for item in image_comparisons[:100]]) + "\n")
    else:
        print("Starting KDTree neighbor search")
        index_vectors = np.vstack(index_vectors)
        print(index_vectors.shape)
        tree = KDTree(index_vectors, leaf_size=75)
        print("Tree generation done")
        test_vectors = np.vstack(test_vectors)
        index_list = []
        print(test_vectors.shape)
        start_point = 0
        current_index_list = []
        if os.path.exists("indices.p"):
            start_point, current_index_list = pickle.load(open("indices.p"), "rb")
        for i in tqdm(range(start_point, len(test_vectors))):
            index_list.append(tree.query([test_vectors[i]], k=100, return_distance=False))
            if i % 1000 == 0:
                pickle.dump((i + 1, index_list), open("indices.p", "wb"))
        print(len(index_list))
        with open(args.submission_file, "w") as submission:
            # First line of submission based on Landmark Retrieval challenge formatting guidelines
            submission.write("id,images\n")
            for i in range(len(index_list)):
                submission.write(test_ids[i]
                                 + ","
                                 + " ".join([index_ids[index] for index in np.squeeze(index_list[i]).tolist()])+ "\n")

