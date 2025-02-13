from src.text_data_utils.bert_embeddings import BertFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import joblib

def predict_nsfw(df, model_path, extractor, threshold=0.85, text_column='prompt', save_path=None):
    """
    Predicts NSFW labels for a new DataFrame using a pre-trained SVM model.

    Parameters:
    - new_df (pd.DataFrame): DataFrame with new data (must contain the text column).
    - model_path (str): Path to the saved SVM model.
    - extractor (BertFeatureExtractor): Pre-trained BERT feature extractor for generating embeddings.
    - threshold (float): Probability threshold for classifying as NSFW (default is 0.85).
    - text_column (str): Column name containing the text to be classified (default is 'prompt').

    Returns None
    - saves the new df
    """
    df['nsfw'] = np.nan
    df['contains_person'] = np.nan
    
    # get embeddings
    new_embeddings = extractor.transform(df, text_column=text_column)
    
    #load model
    model = joblib.load(model_path)
    new_predictions_prob = model.predict_proba(new_embeddings)
    new_predictions = np.where(new_predictions_prob[:, 1] > threshold, 1, 0)
    
    df['nsfw_prediction'] = new_predictions
    df.to_csv(save_path, index=False)
    # return df

if __name__ == "__main__":
    jdb_fp = '/home/ginger/code/gderiddershanghai/Diffusion-Face/data/text_data/JDB_Face_metadata (1).json'
    save_path = '/home/ginger/code/gderiddershanghai/Diffusion-Face/data/text_data/JDB_Face.csv'
    model_path = '/home/ginger/code/gderiddershanghai/Diffusion-Face/weights/svm_model_weights_nsfw.pkl'
    df = pd.read_json(jdb_fp)
    extractor = BertFeatureExtractor()
    