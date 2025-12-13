## 📄 README.md: DeepSeek-OCR FastAPI Service

This project packages the DeepSeek-OCR model into a FastAPI web service, specifically addressing model loading, data type, and configuration issues encountered when running the model locally on a GPU.

### Prerequisites

* NVIDIA GPU with CUDA installed
* Python 3.10+
* `git`

---

### 🚀 Setup and Local Development (Virtual Environment)

This section details the **non-standard setup** required to apply necessary patches and ensure correct model loading.

#### 1. Clone the Project Structure

Clone the specific `test1` branch and set up the `OCR` folder using sparse checkout.

```bash
# Clone the repository (using sparse checkout for efficiency)
git clone --depth 1 --branch test1 --filter=blob:none --no-checkout https://github.com/shubhammatta/DeekSeek-OCR-Docker
cd DeekSeek-OCR-Docker
git sparse-checkout set OCR
git checkout test1

# Navigate to the project root
cd OCR


2. Create the Local Model Structure (Deep Copy)Standard cloning creates symbolic links that bypass the patches. We must perform a deep copy of the model files into the ./DeepSeek-OCR-Local/DeepSeek-OCR folder to ensure the corrected code is loaded.Bash


# Clone the DeepSeek-OCR model into a temporary directory


mkdir -p ./DeepSeek-OCR-Local/DeepSeek-OCR

git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR ./DeepSeek-OCR-Local/DeepSeek-OCR

# Use 'cp -rL' (dereference) to copy the physical files into the final local directory

# Clean up the temporary folder
3. Apply Critical Patches to Local Model FilesDue to known compatibility issues with bfloat16 and custom model identification, two files in the new ./DeepSeek-OCR-Local folder must be overwritten with the patched versions provided in the main OCR folder.Bash# 1. Overwrite modeling_deepseekocr.py (Fixes 'Half and Float' data type error)

cp modeling_deepseek.py ./DeepSeek-OCR-Local/DeepSeek-OCR/modeling_deepseekocr.py

# 2. Overwrite config.json (Fixes 'Unrecognized model' loading error)
cp config.json ./DeepSeek-OCR-Local/DeepSeek-OCR/config.json

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt


5. Run the Uvicorn ServerStart the API server. The application will load the patched model from the local directory specified in server.py

uvicorn server:app --host 0.0.0.0 --port 8000 --reload


🧪 Testing the API

Once the server reports Application startup complete, you can test the service:Documentation (Swagger UI): Access the live docs at http://0.0.0.0:8000/docs.Inference Endpoint: Use the /ocr endpoint to upload an image and receive the extracted markdown text.

