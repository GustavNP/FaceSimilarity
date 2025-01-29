import os
import csv
import pandas as pd




dissimilarity_file = "./similarity_score_files/similarity_scores_Facenet512_yunet_VGGFace-200k-rs-36-test-set.csv"

dissimilarity_df = pd.read_csv(dissimilarity_file, sep=';', header=None)
dissimilarity_df.columns = ['Image_1_Filename', 'Image_2_Filename', '2', 'Dissimilarity_Score', '4', '5', '6']
dissimilarity_df = dissimilarity_df.drop(['2','4','5','6'], axis=1) # not needed
dissimilarity_df.columns = ['Image_1_Filename', 'Image_2_Filename', 'Dissimilarity_Score']
print(dissimilarity_df)



# quality_files = [
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFR-SPECIFIC_FINAL_1-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFR-SPECIFIC_FINAL_2-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFR-SPECIFIC_FINAL_3-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFR-SPECIFIC_FINAL_4-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFR-SPECIFIC_FINAL_5-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFR-SPECIFIC_FINAL_6-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFR-SPECIFIC_FINAL_7-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFR-SPECIFIC_FINAL_8-Test_set-VGGFace200k.csv"
# ]



# quality_files = [
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-BROAD_DEFAULT_1-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_1-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_2-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_3-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_4-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_5-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_6-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_7-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_8-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_9-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_10-All-Test_set-VGGFace200k.csv",
# ]

# quality_files = [
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_11-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_12-All-Test_set-VGGFace200k.csv",
# ]



# quality_files = [
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_13-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_14-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_15-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_16-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_17-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_18-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_19-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_20-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_21-All-Test_set-VGGFace200k.csv",
#     "C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS-RFC-SPECIFIC_FINAL_22-All-Test_set-VGGFace200k.csv",
# ]

# quality_files = [
#     "C:/Users/admin/source/repos/RandomForestUQS/predicted_UQS_files/RFR-All/Predicted-UQS-RFR-BROAD_DEFAULT_1-All-Test_set-VGGFace200k.csv",
#     "C:/Users/admin/source/repos/RandomForestUQS/predicted_UQS_files/RFR-Top-10/Predicted-UQS-RFR-SPECIFIC_FINAL_6-Top-10-Test_set-VGGFace200k.csv",
#     "C:/Users/admin/source/repos/RandomForestUQS/predicted_UQS_files/RFR-Top-15/Predicted-UQS-RFR-SPECIFIC_FINAL_6-Top-15-Test_set-VGGFace200k.csv",
#     "C:/Users/admin/source/repos/RandomForestUQS/predicted_UQS_files/RFR-Top-20/Predicted-UQS-RFR-SPECIFIC_FINAL_6-Top-20-Test_set-VGGFace200k.csv",
#     "C:/Users/admin/source/repos/RandomForestUQS/predicted_UQS_files/RFR-Top-24/Predicted-UQS-RFR-SPECIFIC_FINAL_6-Top-24-Test_set-VGGFace200k.csv",
# ]



quality_files = []

for root, dirs, files in os.walk("C:/Users/admin/source/repos/RandomForestUQS/predicted_UQS_files/RFR-Top-20"):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            quality_files.append(file_path)








for quality_file in quality_files:
    quality_df = pd.read_csv(quality_file, sep=';')
    print(quality_df.describe())
    print(quality_df.dtypes)

    uqs_dictionary = pd.Series(quality_df["UQS"].values, index=quality_df['Filename']).to_dict()
    dissimilarity_df["UQS"] = dissimilarity_df.apply(lambda row: min(uqs_dictionary[row['Image_1_Filename']], uqs_dictionary[row['Image_2_Filename']]), axis=1)
    print(dissimilarity_df)

    score_pairs_df = dissimilarity_df[['Dissimilarity_Score', "UQS"]]
    model_name = quality_file.split('-Test')[0].split('UQS-')[1]
    print(model_name)
    score_pairs_df.to_csv(f"dissimilarity_and_quality_score_pairs/RFR-Top-20/dissimilarity_pairs_for_EDC_Facenet512_03_Yunet_01_and_OFIQ_{model_name}-VGGFace200k-rs-36-test-set.csv", sep=',', header=False, index=False)
    dissimilarity_df = dissimilarity_df.drop(columns=["UQS"])

