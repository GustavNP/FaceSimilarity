from deepface import DeepFace
import os
import csv
import pandas as pd

# based on:
# https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/
# Author: Gustav Nilsson Pedersen - s174562@dtu.dk
# BASED ON CODE WRITTEN IN PREVIOUS COURSE

model_name_use = "Facenet512"
detector_name_use = "yunet"

dataset_name = "test-images-data-set"

root_directory = "test-images"


os.environ["yunet_score_threshold"] = "0.1"

persons = {}


# assumes flat sctructure where filenames are prefixed with person identity. .jpg images only
for root, dirs, files in os.walk(root_directory): # then actually add the file paths
    for file in files:
        if file.lower().endswith('.jpg'):
            person_id = file.split('-')[0]
            if person_id not in persons:
                persons[person_id] = []
            file_path = os.path.join(root, file)
            persons[person_id].append(file_path.replace("\\", "/"))
            print(file_path)



# ============== Compute embeddings =====================

embedding_files = []
embedCounter = 0
for person_name, image_paths in persons.items():
    # print("=======================================")
    # print(person_name)
    output__person_file = f".\\embedding-person-files\\embeddings_{model_name_use}_{detector_name_use}_{dataset_name}_{person_name}.csv"
    embedding_files.append(output__person_file)
    with open(output__person_file, 'w', newline='') as output_csv:
        writer = csv.writer(output_csv, delimiter=';')
        for i in range(0, len(image_paths)): # for each image
            embedding_objs = DeepFace.represent(img_path = image_paths[i], enforce_detection=False, model_name=model_name_use, detector_backend=detector_name_use)
            embedCounter += 1
            if(embedCounter % 100 == 1):
                print('computing embeddings for image number:')
                print(embedCounter)
            embedding_best_face = []
            max_confidence = 0
            for k in range(0, len(embedding_objs)): # check all faces detected and only select the one with the highest confidence (there should only be one face in the image)
                confidence = embedding_objs[k]["face_confidence"]
                if confidence > max_confidence:
                    max_confidence = confidence
                    embedding_best_face = embedding_objs[k]["embedding"]

            # print(maxConfidence)
            # print(len(embeddingBestFace))
            if max_confidence > 0:
                # print("Is face.")
                row = [person_name] + [image_paths[i]] + [max_confidence] + embedding_best_face
                writer.writerow(row)
            
    print(f"Done computing embeddings for person: {person_name}")




output_file_base = f".\\embedding-combined-files\\embeddings_{model_name_use}_{detector_name_use}_{dataset_name}.csv"
output_file = output_file_base

# If the file already exists, create unique name
counter = 1
while os.path.exists(output_file):
    file_name, file_extension = os.path.splitext(output_file_base)
    output_file = f"{file_name}_{counter}{file_extension}"
    counter += 1

# Read and concatenate all CSV files into one DataFrame
combined_df = pd.concat([pd.read_csv(file, header=None) for file in embedding_files], ignore_index=True)
# write to csv
combined_df.to_csv(output_file, index=False, header=False)


print("done computing embeddings")
