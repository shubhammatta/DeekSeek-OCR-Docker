import os
import shutil
import tempfile
import time
from typing import List, Union

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import core logic from the other file
# IMPORTANT: Ensure your core script is named 'deepseek_ocr_core.py'
from deepseek_ocr_core import initialize_model, ocr_from_pdf, ocr_from_image_path

# --- Pydantic Response Model (Optional but Recommended) ---
class OCRResponse(BaseModel):
    """Defines the structure of the JSON response."""
    filename: str
    page_count: int
    execution_time_seconds: float
    extracted_content: List[str]

# --- Global Initialization ---
# This dictionary will store the model and tokenizer objects
GLOBAL_MODEL = {"tokenizer": None, "model": None, "initialized": False}

try:
    TOKENIZER, MODEL = initialize_model()
    if MODEL:
        GLOBAL_MODEL["tokenizer"] = TOKENIZER
        GLOBAL_MODEL["model"] = MODEL
        GLOBAL_MODEL["initialized"] = True
except Exception as e:
    print(f"FATAL: Model initialization failed during startup. {e}")
    # The app will start but endpoints will return 503

# --- FastAPI App Setup ---
app = FastAPI(
    title="DeepSeek-OCR API Service",
    description="A service to convert PDF and Image files to markdown text using a self-hosted DeepSeek-OCR model.",
    version="1.0.0"
)

# --- Middleware/Error Handling ---
@app.on_event("startup")
async def startup_event():
    if not GLOBAL_MODEL["initialized"]:
        print("--- WARNING: Model not initialized. API endpoints will return 503. ---")

@app.get("/")
def read_root():
    return {"status": "Service Running", "model_loaded": GLOBAL_MODEL["initialized"]}

# --- OCR Endpoint ---
@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(
    file: UploadFile = File(..., description="PDF or image file to process (jpg, png, pdf)."),
    prompt: str = "<image>\nConvert the document to markdown."
):
    """
    Accepts a single PDF or image file, performs OCR using the DeepSeek-OCR model, 
    and returns the extracted markdown text.
    """
    if not GLOBAL_MODEL["initialized"]:
        raise HTTPException(
            status_code=503,
            detail="OCR model service is unavailable or failed to initialize. Check server logs."
        )

    # 1. Check file type
    filename = file.filename
    # Handle files with no extension gracefully
    if '.' not in filename:
        raise HTTPException(status_code=400, detail="File has no recognizable extension.")
        
    file_extension = filename.split('.')[-1].lower()
    
    ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'tiff', 'bmp']
    ALLOWED_PDF_EXTENSIONS = ['pdf']
    
    if file_extension not in ALLOWED_IMAGE_EXTENSIONS + ALLOWED_PDF_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{file_extension}. Must be PDF or a common image format."
        )
    
    # 2. Save the uploaded file locally (CRITICAL BRIDGE)
    # The DeepSeek-OCR core expects a file path, so we must save the UploadFile stream.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, filename)
        
        try:
            # Use file.file (the raw SpooledTemporaryFile) with shutil.copyfileobj for efficient streaming
            file.file.seek(0) # Ensure pointer is at the start (in case of prior validation)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file internally: {e}")
        finally:
            # Close the UploadFile stream object explicitly
            await file.close()

        # 3. Call the appropriate OCR function
        start_time = time.time()
        
        try:
            if file_extension in ALLOWED_IMAGE_EXTENSIONS:
                extracted_texts = [ocr_from_image_path(
                    temp_file_path, 
                    GLOBAL_MODEL["tokenizer"], 
                    GLOBAL_MODEL["model"], 
                    prompt
                )]
                
            elif file_extension in ALLOWED_PDF_EXTENSIONS:
                extracted_texts = ocr_from_pdf(
                    temp_file_path, 
                    GLOBAL_MODEL["tokenizer"], 
                    GLOBAL_MODEL["model"], 
                    prompt
                )
                
            else:
                # Should not be reached
                extracted_texts = ["Internal error: File type not recognized."]
                
        except Exception as e:
            # Catch errors from the core OCR script (e.g., unexpected PyMuPDF error)
            print(f"OCR Processing Error: {e}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed due to model error: {e}")

        end_time = time.time()
        
        # 4. Return the result
        return OCRResponse(
            filename=filename,
            page_count=len(extracted_texts),
            execution_time_seconds=round(end_time - start_time, 2),
            extracted_content=extracted_texts
        )

# --- End of Files ---