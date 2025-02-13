from src.text_data_utils.bert_embeddings import BertFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
import joblib



# if __name__ == "__main__":
    print('testing imports')
    fp = '/home/ginger/code/gderiddershanghai/Diffusion-Face/data/text_data/cleaned_DFDB_Face_metadata.csv'
    df = pd.read_csv(fp)

    # import Feature Extractor
    extractor = BertFeatureExtractor()
    # Get Mean Text Embeddings
    train_embeddings = extractor.transform(df, text_column='prompt')

    y_train = df['nsfw'].values
    X_train, X_val, y_train, y_val = train_test_split(train_embeddings, 
                                                      y_train, 
                                                      test_size=0.15, 
                                                      random_state=99)

    # Train SVM - safe weights
    C = 1
    save_fp = f'/home/ginger/code/gderiddershanghai/Diffusion-Face/weights/nsfw_weights_{C}.pkl'
    svm_model = SVC(kernel='rbf', C=C, gamma='scale', random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    y_val_pred = svm_model.predict_proba(X_val)

    thres = 0.85
    y_val_pred = np.where(y_val_pred[:, 1] > thres, 1, 0)
    y_val = np.where(y_val[:, 1] > thres, 1, 0)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    print("Val Acc: ", val_accuracy)
    print(classification_report(y_val, y_val_pred))
    
    # save weights
    joblib.dump(svm_model, save_fp)

    #####################################################
    C = 50
    save_fp = f'/home/ginger/code/gderiddershanghai/Diffusion-Face/weights/nsfw_weights_{C}.pkl'
    svm_model = SVC(kernel='rbf', C=C, gamma='scale', random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    y_val_pred = svm_model.predict_proba(X_val)

    thres = 0.85
    y_val_pred = np.where(y_val_pred[:, 1] > thres, 1, 0)
    y_val = np.where(y_val[:, 1] > thres, 1, 0)
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print("Val Acc: ", val_accuracy)
    print(classification_report(y_val, y_val_pred))
    
    # save weights
    joblib.dump(svm_model, 'svm_model_weights.pkl')
    print(f"Model weights saved to 'svm_model_weights_c{C}.pkl'")
    
    
    C = 95
    save_fp = f'/home/ginger/code/gderiddershanghai/Diffusion-Face/weights/nsfw_weights_{C}.pkl'
    svm_model = SVC(kernel='rbf', C=C, gamma='scale', random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    y_val_pred = svm_model.predict_proba(X_val)

    thres = 0.85
    y_val_pred = np.where(y_val_pred[:, 1] > thres, 1, 0)
    y_val = np.where(y_val[:, 1] > thres, 1, 0)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print("Val Acc: ", val_accuracy)
    print(classification_report(y_val, y_val_pred))
    
    # save weights
    joblib.dump(svm_model, 'svm_model_weights.pkl')
    print(f"Model weights saved to 'svm_model_weights_c{C}.pkl'")