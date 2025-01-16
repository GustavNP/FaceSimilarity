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
dataset_name = "VGGFace-200k-train-431-502"
# dataset_name = "test-few-folders"
# dataset_name = "LFW"

# root_directory = 'C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-51-180'
# root_directory = 'C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-181-304'
# root_directory = 'C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-305-430'
root_directory = 'C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-431-502'
# root_directory = 'C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders'
# root_directory = 'C:\\Users\\admin\\Desktop\\test-few-folders'
# root_directory = 'C:\\Users\\admin\\Downloads\\lfw\\lfw'
# root_directory = 'C:\\Users\\admin\\Desktop\\lfw_sample'

os.environ["yunet_score_threshold"] = "0.1"


persons = {}

# assumes every image of a person is stored in a specific folder for that person (is set to only look for .jpg right now)
for person_directory in os.listdir(root_directory):
    persons[person_directory] = []
    person_path = os.path.join(root_directory, person_directory)
    for filename in os.listdir(person_path):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(person_path, filename)
            persons[person_directory].append(file_path.replace("\\", "/"))
            print(file_path)

# Check how many
# count = 0
# for key, values in identities.items():
#     print(key)
#     print(len(values))
#     for i in range(0, len(values)):
#         # print(values[i])
#         count += 1

# print(len(identities))
# print(count)



# ============== Compute embeddings =====================

embedding_files = []
embedCounter = 0
for person_name, image_paths in persons.items():
    # print("=======================================")
    # print(person_name)
    output__person_file = f".\\embedding_files\\embeddings_{model_name_use}_{detector_name_use}_{dataset_name}_{person_name}.csv"
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




output_file_base = f".\\embeddings_{model_name_use}_{detector_name_use}_{dataset_name}.csv"
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

# === old version, I guess... ===
# with open(output_file, 'w', newline='') as output_csv:
#     writer = csv.writer(output_csv, delimiter=';')
#     for identity, imagePaths in embeddings.items():
#         for imagePath, embeddingDict in imagePaths.items():
#             row = [identity] + [imagePath] + [embeddingDict["face_confidence"]] + embeddingDict["embedding"]
#             writer.writerow(row)

print("done computing embeddings")

# ============ Read embeddings from file instead =============

persons = {} # free memory
embeddings = {}
file_to_read_embeddings_from = output_file
with open(file_to_read_embeddings_from) as fileObject:
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
 
# mated_pairs = pd.DataFrame(mated_pairs, columns = ["file_x", "file_y", "embeddings_x", "embeddings_y"])
# # mated_pairs["decision"] = "Yes"

# print(mated_pairs)

# tempt = pd.DataFrame([],  columns = ["file_x", "file_y"])
# df = pd.concat([mated_pairs, tempt]).reset_index(drop = True)


# mated_pairs.file_x = "dataset/"+mated_pairs.file_x
# mated_pairs.file_y = "dataset/"+mated_pairs.file_y

# instances = mated_pairs[["embeddings_x", "embeddings_y"]].values.tolist()
# print(instances)
#resp_obj = DeepFace.verify(instances, model_name = "VGG-Face", distance_metric = "cosine",)
for i in range(0, len(mated_pairs)):
    resp_obj = DeepFace.verify(mated_pairs[i][2]["embedding"], mated_pairs[i][3]["embedding"], model_name = model_name_use, distance_metric = "cosine")
    mated_pairs[i].append(resp_obj)




output_file_base = f".\\similarity_scores_{model_name_use}_{detector_name_use}_{dataset_name}.csv"
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















# # Write all lines to the output file
# with open(output_file, 'w', newline='') as output_csv:
#     writer = csv.writer(output_csv)
#     writer.writerows(all_lines)

# print(f"Combined {len(all_lines)} lines into {output_file}")
