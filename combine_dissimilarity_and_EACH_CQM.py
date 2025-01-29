import os
import csv
import pandas as pd

# def combine_CQM_and_dissimilarity_score(quality_measures_dictionary, dissimilarity_scores):


#     score_pairs_for_EDC = []
#     for i in range(0, len(dissimilarity_scores)):
#         img1 = dissimilarity_scores[i][0]
#         img2 = dissimilarity_scores[i][1]
#         if img1 not in quality_measures or img2 not in quality_measures: # TODO: we should probably fix this before running this script
#             continue
#         quality_score1 = quality_measures[img1]
#         quality_score2 = quality_measures[img2]
#         qs = min(quality_score1, quality_score2) # use lowest quality score for EDC. De facto standard.
#         score_pairs_for_EDC.append([dissimilarity_scores[i][2], qs])

#     print(len(score_pairs_for_EDC))

#     output_file = "./dissimilarity_and_quality_score_pairs/score_pairs_for_EDC_Facenet512_03_Yunet_01_and_Predicted_UQS_Scalar_to_Scalar_VGGFace200k-Test-set-RFR_SPECIFIC-14-OcclusionFeaturesRemoved.csv"
#     with open(output_file, 'w', newline='') as output_csv:
#         writer = csv.writer(output_csv, delimiter=',')
#         for i in range(0, len(score_pairs_for_EDC)):
#             writer.writerow(score_pairs_for_EDC[i])



#     print(f"Done. Saved pairs to {output_file}")


def remove_nan(value, nan_value, value_instead):
    if value == nan_value:
        return value_instead
    else:
        return int(value)







dissimilarity_file = "./similarity_score_files/similarity_scores_Facenet512_yunet_VGGFace-200k-25-percent-of-images-in-train-set.csv"

dissimilarity_df = pd.read_csv(dissimilarity_file, sep=';', header=None)
dissimilarity_df.columns = ['Image_1_Filename', 'Image_2_Filename', '2', 'Dissimilarity_Score', '4', '5', '6']
dissimilarity_df = dissimilarity_df.drop(['2','4','5','6'], axis=1)
dissimilarity_df.columns = ['Image_1_Filename', 'Image_2_Filename', 'Dissimilarity_Score']
print(dissimilarity_df)

quality_file = "./quality_score_files/VGGFace2-200k-all-OFIQ-only-filenames.csv"
quality_df = pd.read_csv(quality_file, sep=';')


print(quality_df.describe())
print(quality_df.dtypes)


quality_df['EyesVisible.scalar'] = quality_df['EyesVisible.scalar'].apply(lambda x: remove_nan(x, '-nan(ind)', 0))
# print(quality_df['EyesVisible.scalar'].unique())


cqm_names = [
    'BackgroundUniformity.scalar',
    'IlluminationUniformity.scalar',
    'LuminanceMean.scalar',
    'LuminanceVariance.scalar',
    'UnderExposurePrevention.scalar',
    'OverExposurePrevention.scalar',
    'DynamicRange.scalar',
    'Sharpness.scalar',
    'CompressionArtifacts.scalar',
    'NaturalColour.scalar',
    'SingleFacePresent.scalar',
    'EyesOpen.scalar',
    'MouthClosed.scalar',
    'EyesVisible.scalar',
    'MouthOcclusionPrevention.scalar',
    'FaceOcclusionPrevention.scalar',
    'InterEyeDistance.scalar',
    'HeadSize.scalar',
    'LeftwardCropOfTheFaceImage.scalar',
    'RightwardCropOfTheFaceImage.scalar',
    'MarginAboveOfTheFaceImage.scalar',
    'MarginBelowOfTheFaceImage.scalar',
    'HeadPoseYaw.scalar',
    'HeadPosePitch.scalar',
    'HeadPoseRoll.scalar',
    'ExpressionNeutrality.scalar',
    'NoHeadCoverings.scalar'
]

# Remove the dissimilarity scores where one of the images is not in the quality score file
quality_filenames = quality_df['Filename'].values
print(len(dissimilarity_df))
dissimilarity_df = dissimilarity_df[dissimilarity_df['Image_1_Filename'].isin(quality_df['Filename'])]
print(len(dissimilarity_df))
dissimilarity_df = dissimilarity_df[dissimilarity_df['Image_2_Filename'].isin(quality_df['Filename'])]
print(len(dissimilarity_df))



print(quality_df)
for cqm in cqm_names:
    cqm_dictionary = pd.Series(quality_df[cqm].values, index=quality_df['Filename']).to_dict()
    dissimilarity_df[cqm] = dissimilarity_df.apply(lambda row: min(cqm_dictionary[row['Image_1_Filename']], cqm_dictionary[row['Image_2_Filename']]), axis=1)
    print(dissimilarity_df)

    score_pairs_df = dissimilarity_df[['Dissimilarity_Score', cqm]]
    # score_pairs_df.to_csv(f"CQM-and-dissimilarity-pairs/cqm_dissimilarity_pairs_for_EDC_Facenet512_03_Yunet_01_and_OFIQ_{cqm}_VGGFace200k-Test-set.csv", sep=',', header=False, index=False)
    score_pairs_df.to_csv(f"CQM-and-dissimilarity-pairs/cqm_dissimilarity_pairs_for_EDC_Facenet512_03_Yunet_01_and_OFIQ_{cqm}_VGGFace200k-25-percent-of-train-set.csv", sep=',', header=False, index=False)
    dissimilarity_df = dissimilarity_df.drop(cqm, axis=1)

