from deepface import DeepFace
import os
import csv
import pandas as pd

# based on:
# https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/

model_name_use = "Facenet512"
detector_name_use = "yunet"

dataset_name = "VGGFace-200k-train"
# dataset_name = "LFW"

root_directory = 'C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders'
# root_directory = 'C:\\Users\\admin\\Downloads\\lfw\\lfw'
# root_directory = 'C:\\Users\\admin\\Desktop\\lfw_sample'

os.environ["yunet_score_threshold"] = "0.1"


# TODO: persons could be removed and just use embeddings instead. I'm pretty sure...
persons = {}
embeddings = {}

# assumes every image of a person is stored in a specific folder for that person (is set to only look for .jpg right now)
for person_directory in os.listdir(root_directory):
    persons[person_directory] = []
    embeddings[person_directory] = {}
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

embedCounter = 0
for person_name, image_paths in persons.items():
    # print("=======================================")
    # print(person_name)
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
            embeddings[person_name][image_paths[i]] = {}
            embeddings[person_name][image_paths[i]]["embedding"] = embedding_best_face
            embeddings[person_name][image_paths[i]]["face_confidence"] = max_confidence
        
    output__person_file = f".\\embedding_files\\embeddings_{model_name_use}_{detector_name_use}_{dataset_name}_{person_name}.csv"
    with open(output__person_file, 'w', newline='') as output_csv:
        writer = csv.writer(output_csv, delimiter=';')
        for image_with_face_path, image_embedding_dictionary in embeddings[person_name].items(): # for each image of the person, where a face was found
            row = [person_name] + [image_with_face_path] + [image_embedding_dictionary["face_confidence"]] + image_embedding_dictionary["embedding"]
            writer.writerow(row)
    print(f"Done computing embeddings for person: {person_name}")
    






embedCounter = 0
for person_name, image_paths in persons.items():
    # print("=======================================")
    # print(person_name)
    output__person_file = f".\\embedding_files\\embeddings_{model_name_use}_{detector_name_use}_{dataset_name}_{person_name}.csv"
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
            
        for image_with_face_path, image_embedding_dictionary in embeddings[person_name].items(): # for each image of the person, where a face was found
            row = [person_name] + [image_with_face_path] + [image_embedding_dictionary["face_confidence"]] + image_embedding_dictionary["embedding"]
            writer.writerow(row)
    print(f"Done computing embeddings for person: {person_name}")








output_file = f".\\embeddings_{model_name_use}_{detector_name_use}_{dataset_name}.csv"
with open(output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=';')
    for identity, imagePaths in embeddings.items():
        for imagePath, embeddingDict in imagePaths.items():
            row = [identity] + [imagePath] + [embeddingDict["face_confidence"]] + embeddingDict["embedding"]
            writer.writerow(row)

print("done computing embeddings")

# ============ Read embeddings from file instead =============

# file_to_read_embeddings_from = "E:/Dokumenter/embeddings_Facenet512_Yunet_01detect_03recog.csv"
# with open(file_to_read_embeddings_from) as fileObject:
#     reader = csv.reader(fileObject, delimiter=';')
#     for row in reader:
#         identity = row.pop(0)
#         imagePath = row.pop(0)
#         embeddings[identity][imagePath] = {}
#         embeddings[identity][imagePath]["face_confidence"] = float(row.pop(0))
#         embeddings[identity][imagePath]["embedding"] = [float(v) for v in row]
#         print(identity)
#         print(imagePath)
#         print(embeddings[identity][imagePath]["face_confidence"])
#         print(embeddings[identity][imagePath]["embedding"])





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



output_file = f".\\similarity_scores_{model_name_use}_{detector_name_use}_{dataset_name}.csv"
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
