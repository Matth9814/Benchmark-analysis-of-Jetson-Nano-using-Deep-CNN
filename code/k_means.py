import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import json

# Algorithm implementation adapted from: https://www.altoros.com/blog/using-k-means-clustering-in-tensorflow/

#print(tf.__version__)

tf.random.set_seed(42)  # For reproducibility

def plotImg(img, title:str):
    plt.figure()
    plt.title(title)
    plt.imshow(img)

clusters_n = 13
iteration_n = 20

points = []
src_path = "./keras_dataset_5000" # root of the dataset
points_n = src_path.split("_")[2]
print(f"### Uploading {points_n} images ###")
for dir in os.listdir(src_path):
    class_path = "/".join([src_path,dir])
    for file in os.listdir(class_path):
        file_path = "/".join([class_path,file])
        img = Image.open(file_path)
        #plotImg(img,"Original")
        img = tf.image.resize(img,[32,32]).numpy().tolist()
        #print(img)
        #plotImg(img.astype('uint8'),"Scaled")
        #plt.show()
        points.append(img)

#print(points)
#print(len(points))

points = tf.constant(points) # Samples
#points = tf.constant(np.random.uniform(0, 10, (points_n, 2))) # Samples
centroids = tf.random.shuffle(points)[:clusters_n].numpy().tolist()
centroids = tf.Variable(centroids)  # Initial centroids

#print(points.shape)
#print(centroids)

# Add 1 dimension to points (at index 0)
points_expanded = tf.expand_dims(points, 0)
#print(points_expanded.shape)
print("### Starting K-means ###")
for step in range(iteration_n):
    # Add 1 dimension to centroids (at index 1)
    centroids_expanded = tf.expand_dims(centroids, 1)
    #print(centroids_expanded.shape)

    # Compute euclidean distance of each centroid from each point
    # distances is a tensor of dimensions (n_cluster, n_points)
    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), [2,3,4])
    #print(distances.shape)

    # For each point return the nearest centroid index in the centroids array
    assignments = tf.argmin(distances, 0)
    #print(assignments)

    # Centroids new position computation
    means = []
    for c in range(clusters_n):
        means.append(tf.reduce_mean( # Compute the new centroid position
                        tf.gather(points, # Returns the coordinates of the points assigned to each cluster given the index in the points tensor
                            tf.reshape( # Reshapes the input tensor into a (1,#points_in_cluster_c) tensor
                                tf.where( # Returns a list of the True elements indexes
                                    tf.equal(assignments, c)  # Returns a bool array that tells for each point if it is assigned to centroid C (True) or not (False)
                                ), [1, -1]
                            )
                        ), axis=[1]
                    )
        )
    #print(len(means))

    new_centroids = tf.concat(means, 0)
    # Update centroids value
    centroids.assign(new_centroids)
    #print(centroids.shape)

#centroid_values, points_values, assignment_values = centroids.numpy(), points.numpy(), assignments.numpy()
print("Iteration:", step + 1)
#plotClusters(points_values, assignment_values, centroid_values)

# Retrieve values using numpy() only once after updating centroids
centroid_values, points_values, assignment_values = centroids.numpy(), points.numpy(), assignments.numpy()
print("## Classes ##")
for i in range(13):
    print(f"Class {i}: {assignments.numpy().tolist().count(i)}")
plt.hist(assignment_values,bins=range(13))
plt.show()