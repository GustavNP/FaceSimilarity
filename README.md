This project can compute embeddings of faces, calculate comparison scores, and combine comparison scores and quality scores as needed for Error vs. Discard Characteristic (EDC) curves in ERCWare.

The code is loosely based on: https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/
The code is based on code written by Gustav Nilsson Pedersen - s174562@dtu.dk IN A PREVIOUS COURSE at the Technical University of Denmark.

test-images folder contains some test images of 2 people.

The scripts do what their names suggest:
    - combine_dissimilarity_and_quality_scores.py
    - compute_dissimilarity_scores.py
    - compute_embeddings.py

embedding-person-files folder is where the embeddings of each image of a person is saved in a single file for that person, when compute_embeddings.py is run.

embedding-combined-files folder is where all embeddings of all people is saved in a single file, when compute_embeddings.py is run.

comparison-score-files folder is where comparison score files are saved when running compute_dissimilarity_scores.py.

comparison-and-quality-score-pairs is where comparison and quality scores for mated sample pairs are saved when running combine_dissimilarity_and_quality_scores.py. Quality files can be stored in quality-score-files. combine_dissimilarity_and_quality_scores.py is currently set up to expect OFIQ score files with all 27 CQMs and the single UQS (both native and scalar).