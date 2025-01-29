import os
import csv

# Initialize an empty list to store all lines
# similarity_file = "./similarity_scores_Facenet512_Yunet_LFW.csv"
# similarity_file = "./similarity_score_files/similarity_scores_Facenet512_yunet_VGGFace-200k-train-431-502.csv"
similarity_file = "./similarity_score_files/similarity_scores_Facenet512_yunet_VGGFace-200k-images-in-test-set.csv"

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
# quality_file = "./LFW-trained_Scalar-extra-removed_Predicted-UQS_LFW.csv"
# quality_file = "./quality_score_files/VGGFace2-200k-all.csv"
# quality_file = "./quality_score_files/Predicted-UQS_Specific_Dataset_Scalar_to_Scalar_VGGFace200k-n431-502.csv"

# quality_file = "./quality_score_files/Predicted-UQS_Test_set-KNeighbors_VGGFace-200k-only-filenames.csv"
# quality_file = "./quality_score_files/Predicted-UQS_Test_set-RFR_SPECIFIC_9_5-Fold-CV-all-features-vggface200k-only-filenames.csv"
# quality_file = "./quality_score_files/Predicted-UQS_Test_set-RFR-Ablation_Top_10_features_VGGFace-200k-only-filenames.csv"
# quality_file = "./quality_score_files/Predicted-UQS_Test_set-RFR-Ablation_Top_15_features_VGGFace-200k-only-filenames.csv"
# quality_file = "./quality_score_files/Predicted-UQS_Test_set-RFR-Ablation_Top_20_features_VGGFace-200k-only-filenames.csv"
# quality_file = "./quality_score_files/VGGFace2-200k-all-OFIQ-only-filenames.csv"
# quality_file = "./quality_score_files/Predicted-UQS_Test_set-RFR-SPECIFIC-14-All-Features-VGGFace200k-only-filenames.csv"
# quality_file = "./quality_score_files/Predicted-UQS_Test_set-RFR-SPECIFIC-15-All-Features-VGGFace200k-only-filenames.csv"
# quality_file = "./quality_score_files/Predicted-UQS_Test_set_RFR-SPECIFIC-14-NoPreprocessing-VGGFace200k-only-filenames.csv"
# quality_file = "./quality_score_files/Predicted-UQS_SPECIFIC-14-OcclusionFeaturesRemoved-only-filenames.csv"
# quality_file = "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-Test_set-RF-CLASSIFIER-Specific-14-All-Features-VGGFace200k.csv"
# quality_file = "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-Test_set-RF-CLASSIFIER-Specific-14-All-Features-low0-20-high80-VGGFace200k.csv"
# quality_file = "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-Test_set-RF-CLASSIFIER-Specific-14-All-Features-low0-10-high80-VGGFace200k.csv"
# quality_file = "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-Test_set-RF-CLASSIFIER-Specific-14-All-Features-low0-30-high80-VGGFace200k.csv"
# quality_file = "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-Test_set-RF-CLASSIFIER-Specific-14-All-Features-low0-40-high60-VGGFace200k.csv"
# quality_file = "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-Test_set-RF-CLASSIFIER-Specific-14-All-Features-low10-40-high60-VGGFace200k.csv"
# quality_file = "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-Test_set-RF-CLASSIFIER-Specific-14-All-Features-low5-40-high60-VGGFace200k.csv"
quality_file = "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-Test_set-RF-CLASSIFIER-Specific-14-All-Features-low10-50-high51-VGGFace200k.csv"





quality_scores = {}
with open(quality_file) as fileObject:
    reader = csv.reader(fileObject, delimiter=';') # Assumes there IS a header line!!
    next(reader) # skip header if OFIQ sharpness algorithm
    for row in reader:
        imagePath = row.pop(0)
        image_name = imagePath.split('/')[-1]

        # # ==== TEMPORARY ADD "aligned_" ====
        # image_name = "aligned_" + image_name

        # quality_score = int(row.pop(28)) # For UQS scalar from OFIQ output file
        quality_score = int(float(row.pop(0))) # For predicted UQS by random forest model
        quality_scores[image_name] = quality_score
        # print(quality_score)

print(len(quality_scores))


score_pairs_for_EDC = []
for i in range(0, len(similarity_scores)):
    img1 = similarity_scores[i][0]
    img2 = similarity_scores[i][1]
    if img1 not in quality_scores: # TODO: we should probably fix this before running this script
        print(img1)
    if img2 not in quality_scores: # TODO: we should probably fix this before running this script
        print(img2)
    if img1 not in quality_scores or img2 not in quality_scores: # TODO: we should probably fix this before running this script
        continue
    quality_score1 = quality_scores[img1]
    quality_score2 = quality_scores[img2]
    qs = min(quality_score1, quality_score2) # use lowest quality score for EDC. De facto standard.
    score_pairs_for_EDC.append([similarity_scores[i][2], qs])

print(len(score_pairs_for_EDC))


# output_file = "./score_pairs_for_EDC_Facenet512_03_Yunet_01_and_LFW-trained_Scalar-extra-removed_Predicted-UQS_LFW.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-KNeighbors.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RFR_SPECIFIC_9_5-Fold-CV-all-features.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RFR-Ablation_Top_10_features.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RFR-Ablation_Top_15_features.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RFR-Ablation_Top_20_features.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-OFIQ_UQS.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RFR_SPECIFIC-14-5-Fold-CV-all-features.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RFR_SPECIFIC-15-5-Fold-CV-all-features.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RFR_SPECIFIC-14-NoPreprocessing-all-features.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RFR_SPECIFIC-14-OcclusionFeaturesRemoved.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RF-CLASSIFIER_SPECIFIC-14-All-Features-low0-20-high80.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RF-CLASSIFIER_SPECIFIC-14-All-Features-low0-10-high80.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RF-CLASSIFIER_SPECIFIC-14-All-Features-low0-30-high80.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RF-CLASSIFIER_SPECIFIC-14-All-Features-low0-40-high60.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RF-CLASSIFIER_SPECIFIC-14-All-Features-low10-40-high60.csv"
# output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RF-CLASSIFIER_SPECIFIC-14-All-Features-low5-40-high60.csv"
output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RF-CLASSIFIER_SPECIFIC-14-All-Features-low10-50-high51.csv"




with open(output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=',')
    for i in range(0, len(score_pairs_for_EDC)):
        writer.writerow(score_pairs_for_EDC[i])



print(f"Done. Saved pairs to {output_file}")


