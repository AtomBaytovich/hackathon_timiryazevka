import torch
from ct import predict_single_file, Config

if __name__ == "__main__":
    file_path = './horizontal_flip (10).jpg' 
    model_path = './models/best_model_auc_1.0000.pth' 

    prediction, confidence = predict_single_file(file_path, model_path)