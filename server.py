import os
import zipfile
import tempfile
import time
import logging
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ct import predict_single_file, Config
import pydicom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Medical Imaging System", description="Система для анализа КТ-изображений с отчётом")

MODEL_PATH = "./models/best_model_auc_1.0000.pth"

if not os.path.exists(MODEL_PATH):
    logger.error(f"Модель не найдена: {MODEL_PATH}")
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

app.mount("/static", StaticFiles(directory="static"), name="static")

class ProcessingResponse(BaseModel):
    message: str
    report_path: str
    report_data: list

@app.get("/")
async def root():
    """Корневой эндпоинт - возвращает фронтенд"""
    return FileResponse('index.html')

@app.post("/process/", response_model=ProcessingResponse)
async def process_zip(files: list[UploadFile] = File(...)):
    """
    Принимает несколько ZIP-архивов, обрабатывает DICOM-файлы, генерирует XLSX-отчёт и возвращает данные.
    """
    try:
        report_data = []
        report_path = f"reports/report_{int(time.time())}.xlsx"
        os.makedirs("reports", exist_ok=True)

        for zip_file in files:
            # Создаём временную директорию для распаковки
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, zip_file.filename)
                with open(zip_path, "wb") as f:
                    f.write(await zip_file.read())

                # Распаковываем ZIP
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Обходим распакованные файлы
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith('.dcm'):
                            dcm_path = os.path.join(root, file)
                            try:
                                start_time = time.time()
                                dicom = pydicom.dcmread(dcm_path)
                                study_uid = dicom.StudyInstanceUID if 'StudyInstanceUID' in dicom else "Unknown"
                                series_uid = dicom.SeriesInstanceUID if 'SeriesInstanceUID' in dicom else "Unknown"

                                prediction, confidence = predict_single_file(dcm_path, MODEL_PATH)
                                pathology = 1 if prediction == "ПАТОЛОГИЯ" else 0
                                time_taken = time.time() - start_time
                                status = "Success"

                                report_data.append({
                                    "path_to_study": dcm_path.replace(temp_dir, zip_file.filename),
                                    "study_uid": study_uid,
                                    "series_uid": series_uid,
                                    "probability_of_pathology": round(confidence, 4),
                                    "pathology": pathology,
                                    "processing_status": status,
                                    "time_of_processing": round(time_taken, 2)
                                })
                            except Exception as e:
                                logger.error(f"Ошибка обработки файла {dcm_path}: {e}")
                                report_data.append({
                                    "path_to_study": dcm_path.replace(temp_dir, zip_file.filename),
                                    "study_uid": "Error",
                                    "series_uid": "Error",
                                    "probability_of_pathology": 0.0,
                                    "pathology": 0,
                                    "processing_status": f"Failure: {str(e)}",
                                    "time_of_processing": 0.0
                                })

        # Генерируем XLSX-отчёт
        df = pd.DataFrame(report_data)
        df.to_excel(report_path, index=False)

        logger.info(f"Отчёт сгенерирован: {report_path}")

        return ProcessingResponse(
            message="Обработка завершена",
            report_path=report_path,
            report_data=report_data
        )

    except Exception as e:
        logger.error(f"Ошибка обработки ZIP: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/download/{report_path:path}")
async def download_report(report_path: str):
    """Скачивание отчёта"""
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=os.path.basename(report_path))
    raise HTTPException(status_code=404, detail="Отчёт не найден")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)