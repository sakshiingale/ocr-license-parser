import os
import json
import fitz  # PyMuPDF
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import logging
import base64
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Union 
import uvicorn
import re 
import datetime # Kept for date parsing in calculation helpers
from dateutil.parser import parse # Kept for date parsing in calculation helpers

# Load Azure OpenAI credentials
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# --- GENERIC CERTIFICATE Data Models ---

class ExtractedCertificateFields(BaseModel):
    """Internal model for data extracted directly by the Vision API."""
    license_number: Optional[str] = None
    license_title: Optional[str] = None
    start_date: Optional[str] = None
    expiry_date: Optional[str] = None
    IsPermanent: Optional[bool] = None
    cost: Optional[Union[str, int]] = None
    
class FinalCertificateResponse(BaseModel):
    """
    The final response schema matching your internal screen/database schema,
    without calculated date fields.
    """
    # Extracted fields, defaulted to 'NA' if None
    license_number: str = "NA" 
    license_title: str = "NA"
    start_date: str = "NA"
    expiry_date: Optional[str] = "NA"
    IsPermanent: Optional[bool] = False
    cost: Optional[Union[str, int]] = "NA" 
    
    # --- Compliance/Internal Fields ---
    application_days: Optional[int] = 0 # Retained as 0 as requested
    upload_file: Optional[bool] = True
    file_number: Optional[str] = "NA"
    physical_location: Optional[str] = "NA"

# --- Calculation Functions (Kept but not used in final mapping) ---
# We keep these definitions because they are referenced in the helper functions, 
# but the main endpoint no longer uses their results.

def calculate_days_remaining(expiry_date_str: str) -> Optional[int]:
    """Calculates the number of days remaining between the expiry date and today."""
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

        
# --- Helper Functions (Standard) ---

def pdf_validator(pdf_path, password=None):
    """
    Extracts text for error checking (password/integrity) and regex fallback.
    """
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
    """Converts ALL pages of the PDF file to a list of PIL Images."""
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
    """Extracts a generic certificate number using regex patterns."""
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

# --- Vision Functions (Core Logic) ---

def process_image_with_vision(images: list):
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT or not AZURE_OPENAI_API_VERSION:
        logging.error("Missing Azure OpenAI credentials for Vision.")
        return None
        
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
  • Instead, infer the specific license type exactly the way it appears in the expected label list or Identify the exact nature of the permission/approval being granted (e.g., *Biomedical Waste Registration*, *Environmental Clearance*, *Factory License*, *PESO Petroleum Storage License*, *DG Set Registration*, *Labour License*, *Legal Metrology – Verification*, etc.).  
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
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=1000,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        logging.info(f"[DEBUG] Vision Raw Response: {content[:100]}...")
        return content
        
    except Exception as e:
        logging.error(f"[ERROR] Vision API call failed: {e}")
        return None

def extract_fields_with_vision(pdf_path):
    images = pdf_to_images(pdf_path)
    if not images:
        return None
    
    first_result = process_image_with_vision(images)
    if not first_result:
        return None
        
    try:
        cleaned_content = first_result.strip()
        json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
        if not json_match:
            logging.error(f"[ERROR] Vision response did not contain a valid JSON object: {cleaned_content[:50]}...")
            return None
            
        parsed_data = json.loads(json_match.group(0))
   
        mapped_data = {
            'license_number': parsed_data.get('license_number'),
            'license_title': parsed_data.get('license_title'),
            'start_date': parsed_data.get('start_date'),
            'expiry_date': parsed_data.get('expiry_date'),
            'IsPermanent': parsed_data.get('IsPermanent', False),
            'cost': parsed_data.get('cost')
        }
        
        # Final IsPermanent Logic Check (handle boolean conversion)
        if mapped_data.get('IsPermanent') in [True, 'true', 'True']:
            mapped_data['IsPermanent'] = True
            mapped_data['expiry_date'] = None
        else:
            mapped_data['IsPermanent'] = False
        
        return ExtractedCertificateFields(**mapped_data).model_dump()
            
    except json.JSONDecodeError as e:
        logging.error(f"[ERROR] Failed to parse vision result as JSON: {e}")
        return None
    except Exception as e:
        logging.error(f"[ERROR] Vision-based extraction failed: {e}")
        return None

# --- FastAPI App and Endpoint ---
app = FastAPI(title="Generic Certificate Extractor API", version="1.0.0")

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

        # 1. Extract Text (Used only for final Regex fallback)
        text = pdf_validator(pdf_path) 
        
        # 2. VISION MODEL PRIMARY EXECUTION
        logging.info("Executing Vision Model for image-based extraction.")
        fields = extract_fields_with_vision(pdf_path) 
        
        # 3. Final Regex check on text 
        if fields and not fields.get('license_number'):
            fields['license_number'] = extract_license_number_from_text(text)
        renewal_days = FinalCertificateResponse.model_fields['application_days'].default
        extracted_expiry_date = fields.get('expiry_date')
        
        # 4. Map Extracted Fields to Final Schema
        
        final_data = FinalCertificateResponse(
            license_number=fields.get('license_number'),
            license_title=fields.get('license_title'),
            start_date=fields.get('start_date'),
            expiry_date=extracted_expiry_date,
            IsPermanent=fields.get('IsPermanent'),
            cost=fields.get('cost'),
            application_days=calculate_days_remaining(extracted_expiry_date),
            upload_file=True,
            file_number="NA",
            physical_location="NA",
        )
        
        # 5. Return the final, complete structure
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
    """Root endpoint with API information"""
    return {
        "message": "Generic Certificate Extractor API (Vision Only)",
        "version": "1.0.0",
        "endpoint": "/api/extract-fields",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
