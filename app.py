# api.py
import os
import json
import fitz  # PyMuPDF
import logging
import base64
import tempfile
import re
import datetime
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union
from dateutil.parser import parse
import uvicorn

# OpenAI client
from openai import OpenAI

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # change if you want gpt-4o-mini or gpt-4o-mini-preview, etc.

if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not set. Make sure to export OPENAI_API_KEY in your environment.")

# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Data models ---
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

# --- Helpers ---
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
        if doc and getattr(doc, "needs_pass", False):
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

# --- Vision & OpenAI interaction (now using OpenAI API Key) ---
def process_image_with_openai(images: list):
    """
    Convert images to base64 data URIs and send them as part of a user message to the OpenAI chat model.
    The model is instructed to return a single JSON object containing the requested fields.
    Note: some OpenAI models accept images directly as structured message content; here we embed base64
    data URIs in the message text so it's robust against client library differences.
    """
    if not OPENAI_API_KEY:
        logging.error("Missing OPENAI_API_KEY for Vision.")
        return None

    try:
        # Prepare a combined user content: system prompt + the images as inline data URIs
        system_prompt = """
You are a data extraction specialist. Analyze the certificate image(s) and extract the following information in JSON format:

- license_number: The primary certificate, license, or registration number (Extract EXACTLY as seen).
- license_title: The official, specific, non-generic title of the license/certificate.
- start_date: The date of issuance or start date (YYYY-MM-DD format if possible).
- expiry_date: The date until which the certificate is valid (YYYY-MM-DD format if possible, or null if permanent).
- IsPermanent: True if the certificate has no expiry date, otherwise false.
- cost: Return the license or registration fee paid (e.g., "2000 INR", "Rs. 6000").

Consolidate data from ALL pages. If a field is not found, return null. Return ONLY valid JSON (a single top-level JSON object) with no extra text.
"""

        # Build a long user string: short instruction plus each image inserted as "Image N: data:<mime>;base64,<b64>"
        image_texts = []
        for i, image in enumerate(images):
            buffered = BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64}"
            # include a small label for each image; the model will see the data URI
            image_texts.append(f"--- IMAGE PAGE {i+1} START ---\n{data_uri}\n--- IMAGE PAGE {i+1} END ---")

        user_message = "Analyze the following images (each included as a base64 data URI). Consolidate into one JSON object.\n\n" + "\n\n".join(image_texts)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Use the OpenAI client chat completions API
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.05
        )

        # Extract content string (works with the modern client response shape)
        content = None
        # Safe access to nested response structure:
        try:
            # older/newer client shapes: prefer first choice -> message -> content
            choices = getattr(resp, "choices", None) or resp.get("choices", None)
            if choices and len(choices) > 0:
                first = choices[0]
                # different SDK shapes:
                if isinstance(first, dict):
                    # dict-like
                    content = first.get("message", {}).get("content")
                else:
                    # object-like
                    content = getattr(first, "message", None)
                    if content:
                        content = getattr(content, "get", lambda k, default=None: None)("content", None)
        except Exception:
            # fallback: try resp.output_text or str(resp)
            content = getattr(resp, "output_text", None) or str(resp)

        if not content:
            # Try other fallback fields
            content = str(resp)

        content = content.strip()
        logging.info(f"[DEBUG] OpenAI Vision Raw Response (truncated): {content[:300]}...")
        return content

    except Exception as e:
        logging.error(f"[ERROR] OpenAI Vision API call failed: {e}", exc_info=True)
        return None

def extract_fields_with_openai(pdf_path):
    images = pdf_to_images(pdf_path)
    if not images:
        return None

    first_result = process_image_with_openai(images)
    if not first_result:
        return None

    try:
        cleaned_content = first_result.strip()
        json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
        if not json_match:
            # Could be newline-delimited or wrapped; attempt a more lenient extraction
            try:
                parsed = json.loads(cleaned_content)
                parsed_data = parsed
            except Exception:
                logging.error(f"[ERROR] OpenAI response did not contain a direct JSON object: {cleaned_content[:200]}...")
                return None
        else:
            parsed_data = json.loads(json_match.group(0))

        mapped_data = {
            'license_number': parsed_data.get('license_number'),
            'license_title': parsed_data.get('license_title'),
            'start_date': parsed_data.get('start_date'),
            'expiry_date': parsed_data.get('expiry_date'),
            'IsPermanent': parsed_data.get('IsPermanent', False),
            'cost': parsed_data.get('cost')
        }

        if mapped_data.get('IsPermanent') in [True, 'true', 'True', 'TRUE']:
            mapped_data['IsPermanent'] = True
            mapped_data['expiry_date'] = None
        else:
            mapped_data['IsPermanent'] = False

        return ExtractedCertificateFields(**mapped_data).model_dump()

    except json.JSONDecodeError as e:
        logging.error(f"[ERROR] Failed to parse OpenAI result as JSON: {e}")
        return None
    except Exception as e:
        logging.error(f"[ERROR] OpenAI-based extraction failed: {e}", exc_info=True)
        return None

# --- FastAPI App ---
app = FastAPI(title="Generic Certificate Extractor API (OpenAI)", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.post("/api/extract-fields", response_model=FinalCertificateResponse)
async def api_extract_fields(pdf: UploadFile = File(...)):
    if not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await pdf.read()
    pdf_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            pdf_path = tmp.name
            tmp.write(content)

        text = pdf_validator(pdf_path)

        logging.info("Executing OpenAI model for image-based extraction.")
        fields = extract_fields_with_openai(pdf_path)

        # Regex fallback for license number if not extracted
        if fields and not fields.get('license_number'):
            fields['license_number'] = extract_license_number_from_text(text)

        extracted_expiry_date = fields.get('expiry_date') if fields else None

        final_data = FinalCertificateResponse(
            license_number=fields.get('license_number') if fields else "NA",
            license_title=fields.get('license_title') if fields else "NA",
            start_date=fields.get('start_date') if fields else "NA",
            expiry_date=extracted_expiry_date if fields else "NA",
            IsPermanent=fields.get('IsPermanent') if fields else False,
            cost=fields.get('cost') if fields else "NA",
            application_days=calculate_days_remaining(extracted_expiry_date) if extracted_expiry_date else 0,
            upload_file=True,
            file_number="NA",
            physical_location="NA",
        )

        return final_data

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
        "message": "Generic Certificate Extractor API (OpenAI)",
        "version": "1.0.0",
        "endpoint": "/api/extract-fields",
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
