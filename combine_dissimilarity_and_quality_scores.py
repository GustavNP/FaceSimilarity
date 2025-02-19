import os
import csv


# Loosely based on:
# https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/
# Author: Gustav Nilsson Pedersen - s174562@dtu.dk
# BASED ON CODE WRITTEN IN PREVIOUS COURSE

comparison_scores_file = "./comparison-score-files/dissimilarity_scores_Facenet512_yunet_test-images-data-set.csv"

comparison_scores = []
with open(comparison_scores_file) as fileObject:
    reader = csv.reader(fileObject, delimiter=';') # Assumes NO header line!!
    for row in reader:
        imagePath1 = row.pop(0)
        imagePath2 = row.pop(0)
        image_name1 = imagePath1.split('/')[-1]
        image_name2 = imagePath2.split('/')[-1]
        comparison_score = row.pop(1)
        comparison_scores.append([image_name1, image_name2, comparison_score])
print(len(comparison_scores))

quality_file = "./quality-score-files/test-images-quality-scores.csv"

quality_scores = {}
with open(quality_file) as fileObject:
    reader = csv.reader(fileObject, delimiter=';') # Assumes there IS a header line!!
    next(reader) 
    for row in reader:
        imagePath = row.pop(0)
        image_name = imagePath.split('/')[-1]

        quality_score = int(row.pop(28)) # For UQS scalar from OFIQ output file
        # quality_score = int(float(row.pop(0))) # For predicted UQS by random forest model
        quality_scores[image_name] = quality_score
        # print(quality_score)

print(len(quality_scores))


score_pairs_for_EDC = []
for i in range(0, len(comparison_scores)):
    img1 = comparison_scores[i][0]
    img2 = comparison_scores[i][1]
    if img1 not in quality_scores: 
        print(img1)
    if img2 not in quality_scores: 
        print(img2)
    if img1 not in quality_scores or img2 not in quality_scores:
        continue
    quality_score1 = quality_scores[img1]
    quality_score2 = quality_scores[img2]
    qs = min(quality_score1, quality_score2) # use lowest quality score for EDC.
    score_pairs_for_EDC.append([comparison_scores[i][2], qs])

print(len(score_pairs_for_EDC))


output_file = "./comparison-and-quality-score-pairs/score_pairs.csv"


with open(output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=',')
    for i in range(0, len(score_pairs_for_EDC)):
        writer.writerow(score_pairs_for_EDC[i])



print(f"Done. Saved pairs to {output_file}")


