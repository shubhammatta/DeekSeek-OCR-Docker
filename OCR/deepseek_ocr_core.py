import os
import shutil
import tempfile
import torch
import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import traceback

# --- Configuration ---
MODEL_NAME = './DeepSeek-OCR-Local/DeepSeek-OCR'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# T4 GPU (Compute 7.5) requires float16 precision for VRAM efficiency.
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

# --- Model Initialization ---
def initialize_model():
    """Loads the DeepSeek-OCR model and tokenizer with T4 compatibility."""
    print(f"Loading DeepSeek-OCR model {MODEL_NAME} to {DEVICE} with dtype: {DTYPE}")
    try:
        # Suppress potential warnings from trust_remote_code usage in a real environment
        # import logging
        # logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            _attn_implementation='eager' # Stability on T4 (compute 7.5)
        )
        # Apply the determined data type for T4/CPU
        model = model.eval().to(DEVICE, dtype=DTYPE)
        print("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model. Check dependencies, GPU status, and ensure sufficient VRAM.")
        print(f"Details: {e}")
        # Return None, None on failure to be handled by API check
        return None, None

# --- Core OCR Function ---
def ocr_from_image_path(
    image_path: str, 
    tokenizer: AutoTokenizer, 
    model: AutoModel, 
    prompt: str = "<image>\nConvert the document to markdown."
) -> str:
    """
    Performs OCR on a single image file (or PDF page image) using the DeepSeek-OCR model.
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    
    # Use a safe, unique temporary directory for the DeepSeek output files
    with tempfile.TemporaryDirectory() as temp_output_dir:
        # Create a unique path for the output file
        temp_output_base = os.path.join(temp_output_dir, os.path.basename(image_path).replace('.', '_'))

        try:
            # DeepSeek's infer method writes results to a file by default
            # We also set eval_mode=True to ensure the output dict is returned reliably (as per GitHub issues)
            results = model.infer(
                tokenizer, 
                prompt=prompt, 
                image_file=image_path, 
                output_path=temp_output_base, 
                save_results=True,
                eval_mode=True, # Critical for getting output text
                base_size=1024, 
                image_size=640, 
                crop_mode=True
            )
            print(f"Got result {results}")
            # The output text is written to a file (usually .md or .txt) and sometimes in the returned dict
            result_text = results.get('text', None)
            
            # Fallback: Read the generated output file if the text is not in the results dict
            if not result_text:
                output_file = f"{temp_output_base}.md"
                if not os.path.exists(output_file):
                    output_file = f"{temp_output_base}.txt" # Check for .txt fallback

                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        result_text = f.read()
                
            return result_text if result_text else "OCR processing completed, but no text was extracted/found in output files."

        except Exception as e:
            traceback.print_exc()
            return f"OCR Inference Error for {image_path}: {e}"

# --- PDF to Text Wrapper ---
def ocr_from_pdf(
    pdf_path: str, 
    tokenizer: AutoTokenizer, 
    model: AutoModel, 
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown, preserving tables."
) -> List[str]:
    """
    Converts PDF pages to temporary images using PyMuPDF (fitz), 
    performs OCR on each image, and returns a list of results.
    """
    if not os.path.exists(pdf_path):
        return [f"Error: PDF file not found at {pdf_path}"]

    extracted_texts = []
    
    # Use a temporary directory for all generated images
    with tempfile.TemporaryDirectory() as temp_image_dir:
        print(f"Processing PDF: {pdf_path}...")
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Higher resolution (300 DPI equivalent) for better OCR
                zoom = 2.0 
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, dpi=300)
                
                # Save the page image to the temporary directory
                temp_image_path = os.path.join(temp_image_dir, f"page_{page_num + 1}.png")
                pix.save(temp_image_path)
                
                print(f"  -> Running OCR on Page {page_num + 1}...")
                
                page_text = ocr_from_image_path(temp_image_path, tokenizer, model, prompt)
                
                extracted_texts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
            pdf_document.close()
                
            return extracted_texts

        except Exception as e:
            return [f"An error occurred during PDF processing: {e}"]

# --- Simple Local Test Usage ---
if __name__ == '__main__':
    # This block is only for testing the core script locally, not used by FastAPI
    tokenizer, model = initialize_model()
    if model:
        print("Core script ready for local testing. Place test files (e.g., test.jpg, test.pdf) in this directory.")
        
        # Test OCR on 1.png
        image_path = "1.png"
        if os.path.exists(image_path):
            print(f"\nRunning OCR on {image_path}...")
            result = ocr_from_image_path(image_path, tokenizer, model)
            print(f"\n--- OCR Result for {image_path} ---")
            print(result)
            print("--- End of Result ---")
        else:
            print(f"\n{image_path} not found in current directory.")