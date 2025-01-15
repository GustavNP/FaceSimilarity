import os
import csv

# Initialize an empty list to store all lines
similarity_file = "./similarity_scores_Facenet512_Yunet_LFW.csv"

similarity_scores = []
with open(similarity_file) as fileObject:
    reader = csv.reader(fileObject, delimiter=';') # Assumes NO header line!!
    for row in reader:
        imagePath1 = row.pop(0)
        imagePath2 = row.pop(0)
        image_name1 = imagePath1.split('/')[-1]
        image_name2 = imagePath2.split('/')[-1]
        similarity_score = row.pop(1)
        similarity_scores.append([image_name1, image_name2, similarity_score])
        #print(similarity_score)
print(len(similarity_scores))



# quality_file = "./LFW_scores.csv"
# quality_file = "./predicted_UQS.csv"
# quality_file = "./Flickr45k-LFW_trained_Predicted_UQS_LFW.csv"
# quality_file = "./Flickr45k-LFW-trained_Scalar_Predicted-UQS_LFW.csv"
# quality_file = "./LFW-trained_Scalar_Predicted-UQS_LFW.csv"
quality_file = "./LFW-trained_Scalar-extra-removed_Predicted-UQS_LFW.csv"


quality_scores = {}
with open(quality_file) as fileObject:
    reader = csv.reader(fileObject, delimiter=';') # Assumes there IS a header line!!
    next(reader) # skip header if OFIQ sharpness algorithm
    for row in reader:
        imagePath = row.pop(0)
        image_name = imagePath.split('/')[-1]
        # quality_score = int(row.pop(28)) # For UQS scalar from OFIQ output file
        quality_score = int(float(row.pop(0))) # For predicted UQS by random forest model
        quality_scores[image_name] = quality_score
        # print(quality_score)

print(len(quality_scores))


score_pairs_for_EDC = []
for i in range(0, len(similarity_scores)):
    img1 = similarity_scores[i][0]
    img2 = similarity_scores[i][1]
    quality_score1 = quality_scores[img1]
    quality_score2 = quality_scores[img2]
    qs = min(quality_score1, quality_score2) # use lowest quality score for EDC. De facto standard.
    score_pairs_for_EDC.append([similarity_scores[i][2], qs])

print(len(score_pairs_for_EDC))


output_file = "./score_pairs_for_EDC_Facenet512_03_Yunet_01_and_LFW-trained_Scalar-extra-removed_Predicted-UQS_LFW.csv"
with open(output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=',')
    for i in range(0, len(score_pairs_for_EDC)):
        writer.writerow(score_pairs_for_EDC[i])





