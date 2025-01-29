from deepface import DeepFace
import os
import csv
import pandas as pd

# based on:
# https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/

model_name_use = "Facenet512"
detector_name_use = "yunet"

# dataset_name = "VGGFace-200k-train-51-180"
# dataset_name = "VGGFace-200k-train-181-304"
# dataset_name = "VGGFace-200k-train-305-430"
# dataset_name = "VGGFace-200k-train-431-502"
# dataset_name = "VGGFace-200k-images-in-test-set"
# dataset_name = "VGGFace-200k-25-percent-of-images-in-train-set"
dataset_name = "VGGFace-200k-rs-36-test-set"
# dataset_name = "test-few-folders"
# dataset_name = "LFW"

os.environ["yunet_score_threshold"] = "0.1"


# embeddings_file = "larger_embedding_files\embeddings_FaceNet512_yunet_VGGFace200k-images-in-test-set.csv"
# embeddings_file = "larger_embedding_files\embeddings_FaceNet512_yunet_VGGFace200k-25-percent-of-images-in-train-set.csv"
embeddings_file = "larger_embedding_files\embeddings_Facenet512_yunet_VGGFace-200k-rs-36-test-set.csv"


embeddings = {}
with open(embeddings_file) as fileObject:
    reader = csv.reader(fileObject, delimiter=';')
    for row in reader:
        identity = row.pop(0)
        if identity not in embeddings:
            embeddings[identity] = {}

        image_path = row.pop(0)
        if image_path not in embeddings:
            embeddings[identity][image_path] = {}

        embeddings[identity][image_path] = {}
        embeddings[identity][image_path]["face_confidence"] = float(row.pop(0))
        embeddings[identity][image_path]["embedding"] = [float(v) for v in row]
        print(identity)
        print(image_path)
        print(embeddings[identity][image_path]["face_confidence"])
        print(embeddings[identity][image_path]["embedding"])


mated_pairs = []
for identity, imagePaths in embeddings.items():
    imagePathsList = list(imagePaths.keys())
    for i in range(0, len(imagePathsList)-1):
        for k in range(i+1, len(imagePathsList)):
            mated_pair = []
            mated_pair.append(imagePathsList[i])
            mated_pair.append(imagePathsList[k])
            mated_pair.append(imagePaths[imagePathsList[i]])
            mated_pair.append(imagePaths[imagePathsList[k]])
            mated_pairs.append(mated_pair)
 

for i in range(0, len(mated_pairs)):
    resp_obj = DeepFace.verify(mated_pairs[i][2]["embedding"], mated_pairs[i][3]["embedding"], model_name = model_name_use, distance_metric = "cosine")
    mated_pairs[i].append(resp_obj)


output_file_base = f".\\similarity_score_files\\similarity_scores_{model_name_use}_{detector_name_use}_{dataset_name}.csv"
output_file = output_file_base

# If the file already exists, create unique name
counter = 1
while os.path.exists(output_file):
    file_name, file_extension = os.path.splitext(output_file_base)
    output_file = f"{file_name}_{counter}{file_extension}"
    counter += 1

with open(output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=';')
    for i in range(0, len(mated_pairs)):
        row = [mated_pairs[i][0]] + [mated_pairs[i][1]] + [mated_pairs[i][4]["verified"]] + [mated_pairs[i][4]["distance"]] + [mated_pairs[i][4]["threshold"]] + [mated_pairs[i][4]["model"]] + [mated_pairs[i][4]["similarity_metric"]]
        writer.writerow(row)