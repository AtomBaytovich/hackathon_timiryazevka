"""
Система классификации КТ ОГК (органов грудной клетки)
Классификация: НОРМА / ПАТОЛОГИЯ (бинарнаяas)
Поддержка PNG/JPEG, early stopping, дополнительные метрики, балансировка классов, mixed precision, adaptive LR
"""

import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm
import warnings
import logging
from pathlib import Path
import json
import hashlib
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast  

warnings.filterwarnings('ignore')
matplotlib.use('Agg')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    # Пути к данным
    DATA_DIR = "./ct_data"
    NORMAL_DIR = "./ct_data/normal"
    PATHOLOGY_DIR = "./ct_data/pathology"
    MODEL_SAVE_PATH = "./models"
    RESULTS_PATH = "./results"
    
    # Параметры обучения
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5  # Для early stopping
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Параметры изображений
    IMG_SIZE = 224
    NUM_SLICES = 32  
    
    # Параметры модели
    MODEL_NAME = 'efficientnet_b0'  
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.3
    
    # Аугментация
    USE_AUGMENTATION = True
    USE_MIXUP = True 
    MIXUP_ALPHA = 0.4
    
    # Валидация
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    KFOLD_SPLITS = 3  

    # Поддерживаемые форматы
    SUPPORTED_EXTENSIONS = ('.dcm', '.png', '.jpg', '.jpeg')
    
    # Другие параметры
    SEED = 42
    NUM_WORKERS = 4
    USE_AMP = True 

# ==================== ОБРАБОТКА ИЗОБРАЖЕНИЙ ====================
class ImageProcessor:
    """Класс для обработки DICOM, PNG и JPEG файлов"""
    
    @staticmethod
    def load_dicom(file_path):
        """Загрузка DICOM файла"""
        try:
            dicom = pydicom.dcmread(file_path)
            return dicom
        except Exception as e:
            logger.error(f"Ошибка при загрузке DICOM: {e}")
            return None
    
    @staticmethod
    def get_pixels_hu(dicom):
        """Преобразование в единицы Хаунсфилда"""
        image = dicom.pixel_array.astype(np.float64)
        
        intercept = dicom.RescaleIntercept if hasattr(dicom, 'RescaleIntercept') else 0
        slope = dicom.RescaleSlope if hasattr(dicom, 'RescaleSlope') else 1
        
        if slope != 1:
            image = slope * image.astype(np.float64)
        image += intercept
        
        image = np.clip(image, -1000, 1000)
        
        return image
    
    @staticmethod
    def window_image(img, window_center, window_width, intercept=0, slope=1):
        """Применение оконной функции для визуализации"""
        img = (img * slope + intercept)
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img
    
    @staticmethod
    def normalize_image(img):
        """Нормализация изображения в диапазон [0, 255]"""
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
        return (img * 255).astype(np.uint8)
        
    @staticmethod
    def preprocess_image(input_data, target_size=(224, 224)):
        """Полная предобработка изображения в зависимости от формата или массива"""
        try:
            if isinstance(input_data, str):
                # Обработка по пути к файлу
                ext = os.path.splitext(input_data)[1].lower()
                
                if ext == '.dcm':
                    # Обработка DICOM
                    dicom = ImageProcessor.load_dicom(input_data)
                    if dicom is None:
                        logger.error(f"Не удалось загрузить DICOM: {input_data}")
                        return None
                    
                    image_hu = ImageProcessor.get_pixels_hu(dicom)
                    if image_hu is None:
                        return None

                    lung_window = ImageProcessor.window_image(image_hu, -600, 1500)
                    mediastinal_window = ImageProcessor.window_image(image_hu, 40, 400)
                    normalized_lung = ImageProcessor.normalize_image(lung_window)
                    normalized_mediastinal = ImageProcessor.normalize_image(mediastinal_window)
                    combined = np.stack([normalized_lung, normalized_mediastinal, normalized_lung], axis=-1)
                    
                    resized = cv2.resize(combined, target_size, interpolation=cv2.INTER_CUBIC)
                    logger.info(f"DICOM processed, shape: {resized.shape}, dtype: {resized.dtype}")
                    return resized.astype(np.uint8)  # Гарантируем uint8
                
                elif ext in ('.png', '.jpg', '.jpeg'):
                    # Обработка растровых изображений
                    image = cv2.imread(input_data, cv2.IMREAD_COLOR)
                    if image is None:
                        logger.error(f"Ошибка при загрузке изображения: {input_data}")
                        return None
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
                    
                    logger.info(f"Image processed, shape: {resized.shape}, dtype: {resized.dtype}")
                    return resized.astype(np.uint8)  # Гарантируем uint8
                
                else:
                    logger.error(f"Неподдерживаемый формат: {ext}")
                    return None
            
            elif isinstance(input_data, np.ndarray):
                # Обработка np.ndarray
                image = input_data
                if len(image.shape) == 2:  # Если одноканальное изображение
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[-1] == 1:
                    image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
                elif len(image.shape) == 3 and image.shape[-1] == 3:
                    pass  # Уже в формате HWC
                else:
                    logger.error(f"Неправильная форма изображения: {image.shape}")
                    return None
                
                # Если изображение уже в нужном размере, не делаем resize
                if image.shape[:2] != target_size:
                    image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
                
                logger.info(f"ndarray processed, shape: {image.shape}, dtype: {image.dtype}")
                return image.astype(np.uint8)  # Гарантируем uint8
            
            else:
                logger.error(f"Неподдерживаемый тип данных: {type(input_data)}")
                return None
        
        except Exception as e:
            logger.error(f"Ошибка предобработки: {e}")
            return None


# ==================== DATASET ====================
class CTDataset(Dataset):
    """Dataset для КТ изображений"""
    
    def __init__(self, data_paths, labels, transform=None, config=Config):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.config = config
        self.processor = ImageProcessor()
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        label = self.labels[idx]
        
        # Обработка изображения с try-except для robustness
        try:
            image = self.processor.preprocess_image(file_path, 
                                                    target_size=(self.config.IMG_SIZE, 
                                                                 self.config.IMG_SIZE))
        except Exception as e:
            logger.warning(f"Пропуск файла {file_path}: {e}")
            image = np.zeros((self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=np.uint8)
        
        if image is None:
            image = np.zeros((self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=np.uint8)
        
        # Применение аугментаций
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

# ==================== MIXUP ====================
def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== АУГМЕНТАЦИИ ====================
def get_train_transforms(config):
    """Аугментации для обучения (расширенные)"""
    return A.Compose([
        A.RandomResizedCrop(size=(config.IMG_SIZE, config.IMG_SIZE), scale=(0.8, 1.0), p=0.5),  # Исправлено: size вместо height/width
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.ElasticTransform(alpha=120, sigma=6, p=0.3),
        A.GridDistortion(p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(config):
    """Трансформации для валидации"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ==================== МОДЕЛИ ====================
class CTClassifier(nn.Module):
    """Классификатор КТ изображений"""
    
    def __init__(self, config):
        super(CTClassifier, self).__init__()
        self.config = config
        
        # Выбор базовой модели
        if config.MODEL_NAME == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(512, config.NUM_CLASSES)
            )
        elif config.MODEL_NAME == 'resnet50':
            self.base_model = models.resnet50(weights='IMAGENET1K_V2')
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(512, config.NUM_CLASSES)
            )
        elif config.MODEL_NAME == 'densenet121':
            self.base_model = models.densenet121(weights='IMAGENET1K_V1')
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(512, config.NUM_CLASSES)
            )
        
    def forward(self, x):
        return self.base_model(x)

# ==================== ОБУЧЕНИЕ ====================
class Trainer:
    """Класс для обучения модели"""
    
    def __init__(self, model, config, train_labels=None):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Оптимизатор и планировщик
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                     lr=config.LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)  # Исправлено: убрано verbose
        
        # Loss функция с весами классов
        if train_labels is not None:
            class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            logger.info(f"Class weights: {class_weights}")  # Добавлено: логирование весов
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.scaler = GradScaler(enabled=config.USE_AMP)
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_ap': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_ap': []
        }
        
    def train_epoch(self, dataloader):
        """Обучение одной эпохи"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # MixUp если включено
            if self.config.USE_MIXUP:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=self.config.MIXUP_ALPHA)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.USE_AMP):
                outputs = self.model(images)
                if self.config.USE_MIXUP:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item() if not self.config.USE_MIXUP else (lam * (predicted == labels_a).sum().item() + (1 - lam) * (predicted == labels_b).sum().item())
            
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 
                              'Acc': f'{100.*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        epoch_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        epoch_ap = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        
        return epoch_loss, epoch_acc, epoch_auc, epoch_ap
    
    def validate(self, dataloader):
        """Валидация модели"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad(), autocast(enabled=self.config.USE_AMP):
            pbar = tqdm(dataloader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 
                                  'Acc': f'{100.*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        epoch_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        epoch_ap = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        
        return epoch_loss, epoch_acc, epoch_auc, epoch_ap, all_preds, all_labels, all_probs
    
    def train(self, train_loader, val_loader, epochs):
        """Полный цикл обучения с early stopping"""
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Обучение
            train_loss, train_acc, train_auc, train_ap = self.train_epoch(train_loader)
            
            # Валидация
            val_loss, val_acc, val_auc, val_ap, _, _, _ = self.validate(val_loader)
            
            # Логирование изменения learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")
            
            # Обновление планировщика по val_loss
            self.scheduler.step(val_loss)
            
            # Проверка изменения LR
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                logger.info(f"Learning rate reduced to: {new_lr:.6f}")
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_auc'].append(train_auc)
            self.history['train_ap'].append(train_ap)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['val_ap'].append(val_ap)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, AUC: {train_auc:.4f}, AP: {train_ap:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
            
            # Сохранение лучшей модели по val_auc
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.save_model(f'best_model_auc_{val_auc:.4f}.pth')
                logger.info(f"Saved best model with AUC: {val_auc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    def save_model(self, filename):
        """Сохранение модели"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        path = os.path.join(self.config.MODEL_SAVE_PATH, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        logger.info(f"Model saved to {path}")
# ==================== ОЦЕНКА МОДЕЛИ ====================
class ModelEvaluator:
    """Класс для оценки модели"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        
    def evaluate(self, test_loader):
        """Полная оценка модели на тестовых данных"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad(), autocast(enabled=self.config.USE_AMP):
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Метрики (расширенные)
        self.print_metrics(all_labels, all_preds, all_probs)
        self.plot_confusion_matrix(all_labels, all_preds)
        self.plot_roc_curve(all_labels, all_probs)
        self.plot_pr_curve(all_labels, all_probs)
        
        return all_preds, all_labels, all_probs
    
    def print_metrics(self, y_true, y_pred, y_probs):
        """Вывод метрик (расширенный)"""
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_true, y_pred, 
                                    target_names=['Normal', 'Pathology']))
        
        auc_score = roc_auc_score(y_true, y_probs)
        ap_score = average_precision_score(y_true, y_probs)
        f1 = f1_score(y_true, y_pred, average=None)
        print(f"\nROC AUC Score: {auc_score:.4f}")
        print(f"Average Precision Score: {ap_score:.4f}")
        print(f"F1 Score (Normal): {f1[0]:.4f}")
        print(f"F1 Score (Pathology): {f1[1]:.4f}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Построение матрицы ошибок"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Pathology'],
                    yticklabels=['Normal', 'Pathology'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(self.config.RESULTS_PATH, 'confusion_matrix.png'))
        plt.close()
    
    def plot_roc_curve(self, y_true, y_probs):
        """Построение ROC кривой"""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        plt.savefig(os.path.join(self.config.RESULTS_PATH, 'roc_curve.png'))
        plt.close()
    
    def plot_pr_curve(self, y_true, y_probs):
        """Построение Precision-Recall кривой"""
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'PR curve (AP = {ap:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.savefig(os.path.join(self.config.RESULTS_PATH, 'pr_curve.png'))
        plt.close()

# ==================== ИНФЕРЕНС ====================
class CTPredictor:
    """Класс для предсказания на новых данных"""
    
    def __init__(self, model_path, config):
        self.config = config
        self.device = config.DEVICE
        self.model = self.load_model(model_path)
        self.processor = ImageProcessor()
        self.transform = get_val_transforms(config)
        
    def load_model(self, model_path):
        """Загрузка обученной модели"""
        try:
            # Разрешаем загрузку объекта Config
            torch.serialization.add_safe_globals([Config])
            
            model = CTClassifier(self.config)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)  # Временно используем weights_only=False
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели из {model_path}: {e}")
            raise
    
    def predict_single(self, file_path):
        """Предсказание для одного файла"""
        # Обработка изображения
        image = self.processor.preprocess_image(file_path, 
                                                target_size=(self.config.IMG_SIZE, 
                                                             self.config.IMG_SIZE))
        if image is None:
            return None, None
        
        # Применение трансформаций
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Предсказание с AMP
        with torch.no_grad(), autocast(enabled=self.config.USE_AMP):
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
        
        class_names = ['НОРМА', 'ПАТОЛОГИЯ']
        prediction = class_names[predicted.item()]
        confidence = probs[0, predicted.item()].item()
        
        return prediction, confidence
    
    def predict_batch(self, folder):
        """Предсказание для папки с файлами"""
        results = []
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(Config.SUPPORTED_EXTENSIONS)]
        
        for file_path in tqdm(files, desc='Processing files'):
            prediction, confidence = self.predict_single(file_path)
            
            if prediction is not None:
                results.append({
                    'file': os.path.basename(file_path),
                    'prediction': prediction,
                    'confidence': confidence
                })
        
        return pd.DataFrame(results)

# ==================== ЗАГРУЗКА ДАННЫХ ====================
class DataDownloader:
    """Класс для загрузки и подготовки данных"""
    
    @staticmethod
    def download_mosmed_data():
        """
        Загрузка данных MosMedData
        Ссылки на датасеты:
        1. COVID-19: https://mosmed.ai/datasets/covid19_1110/
        2. Normal: Нужно будет дополнительно найти
        """
        print("\n" + "="*50)
        print("ИНСТРУКЦИЯ ПО ЗАГРУЗКЕ ДАННЫХ")
        print("="*50)
        print("\n1. MosMedData COVID-19 CT Dataset:")
        print("   Ссылка: https://mosmed.ai/datasets/covid19_1110/")
        print("   - Скачайте архив")
        print("   - Распакуйте в папку ./ct_data/pathology/")
        
        print("\n2. Дополнительные датасеты с нормальными КТ:")
        print("   - LIDC-IDRI: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254")
        print("   - Отфильтруйте только нормальные случаи")
        print("   - Поместите в папку ./ct_data/normal/")
        
        print("\n3. Структура папок должна быть:")
        print("   ./ct_data/")
        print("   ├── normal/       # Изображения без патологий (dcm/png/jpeg)")
        print("   │   └── *.(dcm/png/jpg/jpeg)")
        print("   └── pathology/    # Изображения с патологиями (dcm/png/jpeg)")
        print("       └── *.(dcm/png/jpg/jpeg)")
        
        # Создание структуры папок
        os.makedirs('./ct_data/normal', exist_ok=True)
        os.makedirs('./ct_data/pathology', exist_ok=True)
        
        print("\nПосле загрузки данных запустите prepare_data()")
    
    @staticmethod
    def prepare_data():
        """Подготовка данных для обучения"""
        normal_files = []
        pathology_files = []
        
        # Сканирование папок на все поддерживаемые форматы
        supported_ext = Config.SUPPORTED_EXTENSIONS
        for root, dirs, files in os.walk(Config.NORMAL_DIR):
            for file in files:
                if file.lower().endswith(supported_ext):
                    normal_files.append(os.path.join(root, file))
        
        for root, dirs, files in os.walk(Config.PATHOLOGY_DIR):
            for file in files:
                if file.lower().endswith(supported_ext):
                    pathology_files.append(os.path.join(root, file))
        
        print(f"Найдено нормальных изображений: {len(normal_files)}")
        print(f"Найдено изображений с патологией: {len(pathology_files)}")
        
        # Создание меток
        labels = [0] * len(normal_files) + [1] * len(pathology_files)
        all_files = normal_files + pathology_files
        
        # Перемешивание
        data = list(zip(all_files, labels))
        np.random.shuffle(data)
        all_files, labels = zip(*data)
        
        return list(all_files), list(labels)

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
def main():
    """Главная функция для запуска всего пайплайна"""
    
    # Установка seed для воспроизводимости
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    print("\n" + "="*50)
    print("CT CHEST CLASSIFICATION SYSTEM (Улучшенная версия с балансировкой и AMP)")
    print("="*50)
    
    # 1. Загрузка и подготовка данных
    print("\n[1/5] Подготовка данных...")
    downloader = DataDownloader()
    
    # Проверка наличия данных
    if not os.path.exists(Config.DATA_DIR):
        print("Данные не найдены. Следуйте инструкции:")
        downloader.download_mosmed_data()
        return
    
    # Подготовка данных
    file_paths, labels = downloader.prepare_data()
    
    if len(file_paths) == 0:
        print("Не найдено файлов. Проверьте структуру папок.")
        return
    
    # 2. Разделение на train/test (val будет в k-fold)
    print("\n[2/5] Разделение данных...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        file_paths, labels, test_size=Config.TEST_SPLIT, 
        random_state=Config.SEED, stratify=labels
    )
    
    print(f"Train+Val: {len(X_train_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Test loader (общий)
    test_dataset = CTDataset(X_test, y_test, 
                             transform=get_val_transforms(Config),
                             config=Config)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # 3. K-fold cross-validation для "внимательного" обучения
    print("\n[3/5] K-fold обучение...")
    skf = StratifiedKFold(n_splits=Config.KFOLD_SPLITS, shuffle=True, random_state=Config.SEED)
    fold_histories = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
        print(f"\nFold {fold+1}/{Config.KFOLD_SPLITS}")
        
        X_train = [X_train_val[i] for i in train_idx]
        y_train = [y_train_val[i] for i in train_idx]
        X_val = [X_train_val[i] for i in val_idx]
        y_val = [y_train_val[i] for i in val_idx]
        
        print(f"Train: {len(X_train)} samples")
        print(f"Val: {len(X_val)} samples")
        
        # DataLoaders
        train_dataset = CTDataset(X_train, y_train, 
                                  transform=get_train_transforms(Config) if Config.USE_AUGMENTATION else get_val_transforms(Config),
                                  config=Config)
        val_dataset = CTDataset(X_val, y_val, 
                                transform=get_val_transforms(Config),
                                config=Config)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                                  shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                                shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
        
        # 4. Создание и обучение модели
        model = CTClassifier(Config)
        trainer = Trainer(model, Config, train_labels=y_train)  # Передаем y_train для class_weights
        trainer.train(train_loader, val_loader, Config.EPOCHS)
        
        fold_histories.append(trainer.history)
        best_models.append(trainer.model)  # Или сохраняем путь
        
    # 5. Оценка на тесте (ensemble или лучшая модель)
    print("\n[5/5] Оценка модели на тестовых данных...")

    evaluator = ModelEvaluator(best_models[-1], Config)
    evaluator.evaluate(test_loader)
    
    # Построение графиков обучения (усредненных по folds)
    avg_history = {k: np.mean([h[k] for h in fold_histories], axis=0) for k in fold_histories[0]}
    plot_training_history(avg_history)
    
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*50)
    print(f"Модель сохранена в: {Config.MODEL_SAVE_PATH}")
    print(f"Результаты сохранены в: {Config.RESULTS_PATH}")

def plot_training_history(history):
    """Построение графиков истории обучения (расширенное)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0,0].plot(history['train_loss'], label='Train Loss')
    axes[0,0].plot(history['val_loss'], label='Val Loss')
    axes[0,0].set_title('Model Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Accuracy
    axes[0,1].plot(history['train_acc'], label='Train Accuracy')
    axes[0,1].plot(history['val_acc'], label='Val Accuracy')
    axes[0,1].set_title('Model Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # AUC
    axes[1,0].plot(history['train_auc'], label='Train AUC')
    axes[1,0].plot(history['val_auc'], label='Val AUC')
    axes[1,0].set_title('Model AUC')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('AUC')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # AP
    axes[1,1].plot(history['train_ap'], label='Train AP')
    axes[1,1].plot(history['val_ap'], label='Val AP')
    axes[1,1].set_title('Model Average Precision')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('AP')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    plt.savefig(os.path.join(Config.RESULTS_PATH, 'training_history.png'))
    plt.close()

# ==================== ИНФЕРЕНС ДЛЯ ОДНОГО ФАЙЛА ====================
def predict_single_file(file_path, model_path):
    """
    Функция для быстрого предсказания одного файла
    
    Args:
        file_path: путь к файлу (dcm/png/jpeg)
        model_path: путь к сохраненной модели
    
    Returns:
        prediction: класс (НОРМА/ПАТОЛОГИЯ)
        confidence: уверенность модели
    """
    predictor = CTPredictor(model_path, Config)
    prediction, confidence = predictor.predict_single(file_path)
    
    if prediction is not None:
        print(f"\nРезультат классификации:")
        print(f"Файл: {file_path}")
        print(f"Предсказание: {prediction}")
        print(f"Уверенность: {confidence:.2%}")
        
        # Визуализация
        visualize_prediction(file_path, prediction, confidence)
    else:
        print("Ошибка при обработке файла")
    
    return prediction, confidence

def visualize_prediction(file_path, prediction, confidence):
    """Визуализация предсказания"""
    processor = ImageProcessor()
    image = processor.preprocess_image(file_path)
    
    if image is not None:
        # Для визуализации используем grayscale если возможно
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        plt.figure(figsize=(10, 8))
        plt.imshow(gray, cmap='gray')
        
        # Добавление текста с предсказанием
        color = 'green' if prediction == 'НОРМА' else 'red'
        plt.title(f'Предсказание: {prediction} (Уверенность: {confidence:.2%})', 
                  fontsize=16, color=color, weight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Сохранение результата
        os.makedirs(Config.RESULTS_PATH, exist_ok=True)
        filename = os.path.basename(file_path).rsplit('.', 1)[0] + '_prediction.png'
        plt.savefig(os.path.join(Config.RESULTS_PATH, filename))
        plt.close()

# ==================== ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ====================
class DataAnalyzer:
    """Класс для анализа данных"""
    
    @staticmethod
    def analyze_dataset(data_dir):
        """Анализ датасета"""
        stats = {
            'total_files': 0,
            'normal_count': 0,
            'pathology_count': 0,
            'corrupted_files': [],
            'file_sizes': [],
            'patient_ids': set(),
            'study_dates': set(),
            'modalities': set(),
            'formats': {}
        }
        
        supported_ext = Config.SUPPORTED_EXTENSIONS
        
        # Анализ нормальных изображений
        normal_dir = os.path.join(data_dir, 'normal')
        if os.path.exists(normal_dir):
            for file in os.listdir(normal_dir):
                if file.lower().endswith(supported_ext):
                    file_path = os.path.join(normal_dir, file)
                    ext = os.path.splitext(file)[1].lower()
                    stats['normal_count'] += 1
                    stats['total_files'] += 1
                    stats['formats'][ext] = stats['formats'].get(ext, 0) + 1
                    
                    stats['file_sizes'].append(os.path.getsize(file_path))
                    
                    if ext == '.dcm':
                        try:
                            dicom = pydicom.dcmread(file_path, stop_before_pixels=True)
                            if hasattr(dicom, 'PatientID'):
                                stats['patient_ids'].add(dicom.PatientID)
                            if hasattr(dicom, 'StudyDate'):
                                stats['study_dates'].add(dicom.StudyDate)
                            if hasattr(dicom, 'Modality'):
                                stats['modalities'].add(dicom.Modality)
                        except:
                            stats['corrupted_files'].append(file_path)
        
        # Анализ патологических изображений
        pathology_dir = os.path.join(data_dir, 'pathology')
        if os.path.exists(pathology_dir):
            for file in os.listdir(pathology_dir):
                if file.lower().endswith(supported_ext):
                    file_path = os.path.join(pathology_dir, file)
                    ext = os.path.splitext(file)[1].lower()
                    stats['pathology_count'] += 1
                    stats['total_files'] += 1
                    stats['formats'][ext] = stats['formats'].get(ext, 0) + 1
                    
                    stats['file_sizes'].append(os.path.getsize(file_path))
                    
                    if ext == '.dcm':
                        try:
                            dicom = pydicom.dcmread(file_path, stop_before_pixels=True)
                            if hasattr(dicom, 'PatientID'):
                                stats['patient_ids'].add(dicom.PatientID)
                            if hasattr(dicom, 'StudyDate'):
                                stats['study_dates'].add(dicom.StudyDate)
                            if hasattr(dicom, 'Modality'):
                                stats['modalities'].add(dicom.Modality)
                        except:
                            stats['corrupted_files'].append(file_path)
        
        # Вывод статистики
        print("\n" + "="*50)
        print("АНАЛИЗ ДАТАСЕТА")
        print("="*50)
        print(f"Всего файлов: {stats['total_files']}")
        print(f"Нормальные изображения: {stats['normal_count']}")
        print(f"Изображения с патологией: {stats['pathology_count']}")
        print(f"Поврежденные файлы: {len(stats['corrupted_files'])}")
        print(f"Уникальных пациентов: {len(stats['patient_ids'])}")
        print(f"Диапазон дат исследований: {len(stats['study_dates'])} уникальных дат")
        print(f"Модальности: {', '.join(stats['modalities'])}")
        print(f"Форматы файлов: {stats['formats']}")
        
        if stats['file_sizes']:
            avg_size = np.mean(stats['file_sizes']) / (1024 * 1024)  # В МБ
            print(f"Средний размер файла: {avg_size:.2f} МБ")
        
        # Визуализация распределения
        if stats['total_files'] > 0:
            plt.figure(figsize=(15, 5))
            
            # График распределения классов
            plt.subplot(1, 3, 1)
            labels = ['Норма', 'Патология']
            sizes = [stats['normal_count'], stats['pathology_count']]
            colors = ['#2ecc71', '#e74c3c']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Распределение классов')
            
            # График форматов
            plt.subplot(1, 3, 2)
            formats = list(stats['formats'].keys())
            counts = list(stats['formats'].values())
            plt.bar(formats, counts, color='#3498db')
            plt.title('Распределение форматов')
            plt.xlabel('Формат')
            plt.ylabel('Количество')
            
            # График размеров файлов
            if stats['file_sizes']:
                plt.subplot(1, 3, 3)
                sizes_mb = [s/(1024*1024) for s in stats['file_sizes']]
                plt.hist(sizes_mb, bins=30, color='#3498db', edgecolor='black')
                plt.xlabel('Размер файла (МБ)')
                plt.ylabel('Количество файлов')
                plt.title('Распределение размеров файлов')
            
            plt.tight_layout()
            plt.show()
        
        return stats

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    # Запуск основного пайплайна
    main()
    
    # Дополнительный анализ 
    # analyzer = DataAnalyzer()
    # analyzer.analyze_dataset(Config.DATA_DIR)
    
    # Предсказание для одного файла
    # predict_single_file("path/to/your/file.dcm", "models/best_model.pth")