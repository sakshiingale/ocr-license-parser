import os
import json
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import logging
import time
import base64
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Union
import uvicorn
import re
import datetime
from dateutil.parser import parse
import uuid

# NEW: token counting
try:
    import tiktoken
except Exception:
    tiktoken = None
    logging.warning("[WARN] tiktoken not installed. Install with `pip install tiktoken` for fallback token counting.")

# Load Azure OpenAI credentials
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Token log path (server-side)
API_TOKEN_LOG = "api_token_logs.jsonl"

# --- GENERIC CERTIFICATE Data Models ---
class ExtractedCertificateFields(BaseModel):
    license_number: Optional[str] = None
    license_title: Optional[str] = None
    start_date: Optional[str] = None
    expiry_date: Optional[str] = None
    IsPermanent: Optional[bool] = None
    cost: Optional[Union[str, int]] = None

class FinalCertificateResponse(BaseModel):
    license_number: str = "NA"
    license_title: str = "NA"
    start_date: str = "NA"
    expiry_date: Optional[str] = "NA"
    IsPermanent: Optional[bool] = False
    cost: Optional[Union[str, int]] = "NA"
    application_days: Optional[int] = 0
    upload_file: Optional[bool] = True
    file_number: Optional[str] = "NA"
    physical_location: Optional[str] = "NA"

# --- helpers ---
def calculate_days_remaining(expiry_date_str: str) -> Optional[int]:
    if not expiry_date_str or expiry_date_str == "NA" or expiry_date_str.upper() == "NULL":
        return None
    try:
        expiry_date = parse(expiry_date_str).date()
        today = datetime.date.today()
        delta = expiry_date - today
        return max(0, delta.days)
    except Exception as e:
        logging.error(f"[DAYS REMAINING ERROR] Could not parse date '{expiry_date_str}': {e}")
        return None

def pdf_validator(pdf_path, password=None):
    logging.info(f"[INFO] Extracting text from PDF (for validation/regex only): {pdf_path}")
    text = []
    doc = None
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        if doc and doc.needs_pass:
            raise RuntimeError('PDF is password protected. Authentication failed.')
        raise RuntimeError(f"PDF validation failed: {e}")

def pdf_to_images(pdf_path, dpi=200):
    logging.info(f"[INFO] Converting ALL PDF pages to images at {dpi} DPI.")
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for index in range(len(pdf_document)):
            page = pdf_document[index]
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data)).convert("RGB")
            images.append(img)
        return images
    except Exception as e:
        logging.error(f"[ERROR] Failed to convert PDF to images: {e}")
        return []

def extract_license_number_from_text(text):
    patterns = [
        r'(?:License\s*No\.?\s*:?\s*|Certificate\s*No\.?\s*:?\s*|Reg\s*No\.?\s*:?\s*)([A-Z0-9\-\/]{5,30})',
        r'(\b[A-Z0-9]{5,15}\d{4,}\b)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            logging.info(f"[INFO] Found generic certificate number via regex: {match.group(1)}")
            return match.group(1)
    return None

# --- token helpers ---
FALLBACK_ENCODING = "cl100k_base"
def _get_encoding_for_model(model_name: str):
    if not tiktoken:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding(FALLBACK_ENCODING)
        except Exception:
            return None

def count_tokens_for_text(text: str, model_name: str = "gpt-4o") -> int:
    enc = _get_encoding_for_model(model_name)
    if enc:
        return len(enc.encode(text))
    # last fallback: estimate by words * 1.3
    return int(max(1, len(text.split()) * 1.3))

# --- Vision Functions (Core Logic) ---
def process_image_with_vision(images: list):
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT or not AZURE_OPENAI_API_VERSION:
        logging.error("Missing Azure OpenAI credentials for Vision.")
        return None, None  # (content, usage)

    try:
        user_content_list = []
        for i, image in enumerate(images):
            buffered = BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64}"
            user_content_list.append({"type": "image_url", "image_url": {"url": data_uri}})

        user_content_list.insert(0, {
            "type": "text",
            "text": "Analyze all pages of the document. Extract the details for the **most important/primary license or certificate** and consolidate all requested fields into a single JSON object."
        })

        messages = [
            {
                "role": "system",
                "content": """
You are a data extraction specialist. Analyze the certificate image(s) and extract the following information in JSON format:

- license_number: The primary certificate, license, or registration number (Extract EXACTLY as seen).

- license_title: Extract the **true regulatory license/certificate type** based on the actual purpose, authority, and nature of permission granted in the document — NOT the decorative heading printed on top.
  
  • DO NOT return generic headings such as "Registration Certificate", "Certificate", "Order", "Form", "Verification", "Environmental Clearance (Cover Page)", "Certificate of Authorization", etc., even if they appear in large/bold text.  
  • Instead, infer the specific license type exactly the way it appears in the expected label list or Identify the exact nature of the permission/approval being granted (e.g., *Biomedical Waste Registration*, *Environmental Clearance*, *Factory License*, *PESO Petroleum Storage License*, *DG Set Registration*, *Labour License*, *Legal Metrology -Verification*, etc.).  
  • Derive the title from regulatory clues such as the issuing department, applicable acts/rules, type of permission (e.g., storage, operation, generator, hazardous material, labour registration, factory registration), category/class, or license purpose.
  • If the text is in a regional language, translate the **meaningful regulatory license type**, not the literal header wording and return in english.
  • The final title must reflect the **actual type of government approval** represented by the document, not its layout heading.  
  • Only return the regulatory license name. If no meaningful license type can be determined, return null.

- start_date: The date of issuance or start date (YYYY-MM-DD format if possible).

- expiry_date: The date until which the certificate is valid (YYYY-MM-DD format if possible, or null if permanent).

- IsPermanent: True if the certificate has no expiry date, otherwise false.

- cost: Return the license or registration fee paid (e.g., "2000 INR", "Rs. 6000").

CRITICAL INSTRUCTIONS:
Consolidate data from ALL pages. 
If a field is not found, use null. 
Extract the EXACT license_number. 
Return ONLY valid JSON with no extra text.

"""
            },
            {
                "role": "user",
                "content": user_content_list,
            }
        ]

        client = openai.AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

        # Call with simple retry logic
        max_attempts = 3
        attempt = 0
        response = None
        while True:
            try:
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.1
                )
                break
            except Exception as e:
                attempt += 1
                logging.warning(f"[RETRY] Vision API call failed on attempt {attempt}: {e}")
                if attempt >= max_attempts:
                    logging.error(f"[ERROR] Vision API call failed after {max_attempts} attempts.")
                    raise
                time.sleep(0.5 * attempt)

        content = response.choices[0].message.content.strip()

        # Try to get usage in a robust way (object or dict)
        usage = None
        try:
            usage = getattr(response, "usage", None)
        except Exception:
            try:
                usage = response.get("usage")
            except Exception:
                usage = None

        return content, {"usage": usage, "messages": messages}

    except Exception as e:
        logging.error(f"[ERROR] Vision API call failed: {e}")
        return None, None

def extract_fields_with_vision(pdf_path):
    images = pdf_to_images(pdf_path)
    if not images:
        return None, None

    first_result, meta = process_image_with_vision(images)
    if not first_result:
        return None, None

    try:
        cleaned_content = first_result.strip()
        json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
        if not json_match:
            logging.error(f"[ERROR] Vision response did not contain a valid JSON object: {cleaned_content[:50]}...")
            return None, meta

        parsed_data = json.loads(json_match.group(0))

        mapped_data = {
            'license_number': parsed_data.get('license_number'),
            'license_title': parsed_data.get('license_title'),
            'start_date': parsed_data.get('start_date'),
            'expiry_date': parsed_data.get('expiry_date'),
            'IsPermanent': parsed_data.get('IsPermanent', False),
            'cost': parsed_data.get('cost')
        }

        if mapped_data.get('IsPermanent') in [True, 'true', 'True']:
            mapped_data['IsPermanent'] = True
            mapped_data['expiry_date'] = None
        else:
            mapped_data['IsPermanent'] = False

        return ExtractedCertificateFields(**mapped_data).model_dump(), meta

    except json.JSONDecodeError as e:
        logging.error(f"[ERROR] Failed to parse vision result as JSON: {e}")
        return None, meta
    except Exception as e:
        logging.error(f"[ERROR] Vision-based extraction failed: {e}")
        return None, meta

# --- FastAPI App and Endpoint ---
app = FastAPI(title="Generic Certificate Extractor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.post("/api/extract-fields")  # removed response_model to allow tokens wrapper
async def api_extract_fields(pdf: UploadFile = File(...)):
    if not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    request_id = str(uuid.uuid4())
    start_time = time.time()
    pdf_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            pdf_path = tmp.name
            tmp.write(await pdf.read())

        text = pdf_validator(pdf_path)  # used for fallback regex
        logging.info(f"[{request_id}] Executing Vision Model for image-based extraction.")
        fields, meta = extract_fields_with_vision(pdf_path)

        # fallback regex if license_number missing
        if fields and not fields.get('license_number'):
            fields['license_number'] = extract_license_number_from_text(text)

        extracted_expiry_date = fields.get('expiry_date') if fields else None
        final_data = FinalCertificateResponse(
            license_number=(fields.get('license_number') if fields else "NA"),
            license_title=(fields.get('license_title') if fields else "NA"),
            start_date=(fields.get('start_date') if fields else "NA"),
            expiry_date=extracted_expiry_date if extracted_expiry_date else "NA",
            IsPermanent=(fields.get('IsPermanent') if fields else False),
            cost=(fields.get('cost') if fields else "NA"),
            application_days=calculate_days_remaining(extracted_expiry_date),
            upload_file=True,
            file_number="NA",
            physical_location="NA",
        )

        # --- TOKEN HANDLING ---
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        model_name = AZURE_OPENAI_DEPLOYMENT or "unknown"

        usage = None
        messages = None
        if meta:
            usage = meta.get("usage")
            messages = meta.get("messages")

        # If API provided usage, prefer it
        if usage:
            # usage may be an object or dict
            try:
                prompt_tokens = int(getattr(usage, "prompt_tokens", usage.get("prompt_tokens", 0)))
            except Exception:
                try:
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                except Exception:
                    prompt_tokens = 0
            try:
                completion_tokens = int(getattr(usage, "completion_tokens", usage.get("completion_tokens", 0)))
            except Exception:
                try:
                    completion_tokens = int(usage.get("completion_tokens", 0))
                except Exception:
                    completion_tokens = 0
            total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens))
        else:
            # Fallback to tiktoken counting of the messages and response string
            # messages -> serialize safely
            try:
                messages_text = json.dumps(messages, default=str)
            except Exception:
                messages_text = str(messages)
            # 'first_result' (raw string) is not returned by extract_fields_with_vision to avoid large memory here,
            # but we can approximate completion tokens by counting only the parsed JSON fields returned to client.
            # Count prompt and the compact result
            prompt_tokens = count_tokens_for_text(messages_text, model_name)
            # Count the *result* payload we will return (final_data as JSON)
            completion_tokens = count_tokens_for_text(json.dumps(final_data.model_dump(), default=str), model_name)
            total_tokens = prompt_tokens + completion_tokens

        end_time = time.time()
        duration_ms = int((end_time - start_time) * 1000)

        # server-side minimal token log (no text)
        server_log = {
            "request_id": request_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "model": model_name,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "duration_ms": duration_ms,
            "status": "success"
        }
        try:
            with open(API_TOKEN_LOG, "a", encoding="utf-8") as flog:
                flog.write(json.dumps(server_log) + "\n")
        except Exception as e:
            logging.warning(f"[TOKEN LOG] Failed to write token log: {e}")

        # Return wrapper with result + tokens metadata so client (extractions.py) can record tokens
        return {
            "result": final_data.model_dump(),
            "tokens": {
                "model": model_name,
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or 0),
            },
            "request_id": request_id,
            "duration_ms": duration_ms
        }

    except RuntimeError as e:
        logging.error(f"[FATAL] PDF Runtime Error: {e}")
        raise HTTPException(status_code=400, detail=f'The PDF could not be processed: {str(e)}')
    except Exception as e:
        logging.error(f"[FATAL] Unexpected Error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f'An unexpected error occurred: {str(e)}')
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)

@app.get("/")
async def root():
    return {
        "message": "Generic Certificate Extractor API (Vision Only)",
        "version": "1.0.0",
        "endpoint": "/api/extract-fields",
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)

