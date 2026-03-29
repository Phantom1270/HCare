import streamlit as st
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import uuid
import hashlib
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import cloudinary
import cloudinary.uploader
import easyocr
import numpy as np
import re
from datetime import datetime
import concurrent.futures
import extra_streamlit_components as stx
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Try Streamlit secrets first (for deployment)
if "MONGO_URI" in st.secrets:
    MONGO_URI = st.secrets["MONGO_URI"]
else:
    MONGO_URI = os.getenv("MONGO_URI")

st.set_page_config(page_title="MedVigilant", page_icon="🛡️", layout="wide")

# Initialize persistent cookie manager
cookie_manager = stx.CookieManager()


# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------

# Cloudinary Config
if "CLOUDINARY_CLOUD_NAME" in st.secrets:
    CLOUD_NAME = st.secrets["CLOUDINARY_CLOUD_NAME"]
    API_KEY = st.secrets["CLOUDINARY_API_KEY"]
    API_SECRET = st.secrets["CLOUDINARY_API_SECRET"]
else:
    CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    API_KEY = os.getenv("CLOUDINARY_API_KEY")
    API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET
)


@st.cache_resource
def init_db():
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client.HCare_DB
    return client, db

client, db = init_db()
cabinet_col = db.cabinet
triage_col = db.triage_history
users_col = db.users
prescriptions_col = db.prescriptions

# ---------------------------------------------------------
# Authentication Logic
# ---------------------------------------------------------
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# Auto-login via Persistent Auth Cookie
auth_cookie = cookie_manager.get(cookie="medvigilant_auth")
if auth_cookie and not st.session_state.logged_in:
    st.session_state.logged_in = True
    st.session_state.username = auth_cookie


if not st.session_state.logged_in:
    st.title("🛡️ MedVigilant - Login")
    
    auth_tabs = st.tabs(["Login", "Sign Up"])
    
    with auth_tabs[0]:
        st.subheader("Login to your account")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", type="primary"):
            if not login_username or not login_password:
                st.warning("Please enter both username and password.")
            else:
                user = users_col.find_one({"username": login_username})
                if user and user.get("password") == hash_password(login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    # Write highly-durable auth cookie (expires in 7 days)
                    from datetime import timedelta
                    expire_date = datetime.now() + timedelta(days=7)
                    cookie_manager.set("medvigilant_auth", login_username, expires_at=expire_date)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
    with auth_tabs[1]:
        st.subheader("Create a new account")
        reg_username = st.text_input("Username", key="reg_user")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Sign Up", type="primary"):
            if not reg_username or not reg_password:
                st.warning("Please fill in all fields.")
            elif reg_password != reg_confirm:
                st.error("Passwords do not match.")
            else:
                existing_user = users_col.find_one({"username": reg_username})
                if existing_user:
                    st.error("Username already exists. Please choose a different one.")
                else:
                    new_user = {
                        "username": reg_username,
                        "password": hash_password(reg_password)
                    }
                    users_col.insert_one(new_user)
                    st.success("Account created successfully! You can now log in.")
    
    st.stop() # Stops execution of the rest of the app if not logged in

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def check_drug_rash_link(drug_name):
    """Hits the OpenFDA Adverse Events API to see if the drug is known to cause rashes."""
    try:
        # Looking for reports where the drug name is mentioned AND the reaction contains rash/erythema/dermatitis/urticaria
        query = f'patient.drug.medicinalproduct:"{drug_name}"+AND+patient.reaction.reactionmeddrapt:("RASH"+OR+"ERYTHEMA"+OR+"DERMATITIS"+OR+"URTICARIA")'
        url = f'https://api.fda.gov/drug/event.json?search={query}&limit=1'
        
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            data = res.json()
            if data.get('results'):
                return True
    except Exception:
        pass
    return False

def get_rxcui(drug_name):
    """Fetch the RxCUI for a given drug name using the RxNorm API."""
    try:
        url = f"https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term={drug_name}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if candidates:
            # RxNorm "score" is not out of 100, it's an arbitrary value.
            # We return the top match. The 70% confidence check will be handled
            # via string similarity in the extraction pipeline.
            return candidates[0].get("rxcui")
    except Exception as e:
        # Silently fail if API limit or error occurs during concurrent processing
        pass
    return None

def check_interactions(new_drug_name):
    """Check if the new drug interacts with anything currently in the cabinet using the live OpenFDA API."""
    if not new_drug_name:
        return False, None
    
    current_drugs = list(cabinet_col.find({"username": st.session_state.username}))
    if not current_drugs:
        return False, None # Nothing to interact with
        
    for existing_drug in current_drugs:
        existing_name = existing_drug.get("name")
        if not existing_name: continue
        
        # Build FDA query checking both directions. 
        # Crucial fix: Use EXACT PHRASE matching for the interaction target to prevent 
        # OpenFDA from returning a label just because it mentions both drugs anywhere in the document.
        q1 = f'(openfda.generic_name:"{new_drug_name}"+AND+drug_interactions:"\\"{existing_name}\\"")'
        q2 = f'(openfda.generic_name:"{existing_name}"+AND+drug_interactions:"\\"{new_drug_name}\\"")'
        query = f'{q1}+OR+{q2}'
        
        url = f'https://api.fda.gov/drug/label.json?search={query}&limit=1'
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    raw_text = data['results'][0].get('drug_interactions', ['High-risk interaction detected.'])[0]
                    
                    # Double check if the text actually mentions the OTHER drug to avoid generic warnings
                    search_target = new_drug_name.lower() if data['results'][0].get('openfda', {}).get('generic_name', [''])[0].lower() == existing_name.lower() else existing_name.lower()
                    if search_target not in raw_text.lower():
                        continue # False positive hit
                    
                    # Clamp string length for UI readability
                    clamped_text = (raw_text[:250] + '...') if len(raw_text) > 250 else raw_text
                    warning_msg = f"Between **{existing_name}** and this new drug (Source: OpenFDA):\n\n{clamped_text}"
                    return True, warning_msg
        except Exception as e:
            st.sidebar.error(f"FDA API Error checking interactions: {e}")
            
    return False, None

@st.cache_resource
def load_model():
    """Load the pre-trained MobileNetV2 strictly for demo classification."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    model.eval()
    
    # Standard ImageNet transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess

@st.cache_resource
def load_ocr_model():
    """Load the EasyOCR English model."""
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return reader

def get_spelling_suggestion(word):
    """Hits the RxNav Spellingsuggestions API to correct OCR mistakes."""
    try:
        url = f"https://rxnav.nlm.nih.gov/REST/spellingsuggestions.json?name={word}"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            suggestions = data.get('suggestionGroup', {}).get('suggestionList', {}).get('suggestion', [])
            if suggestions:
                return suggestions[0]
    except Exception:
        pass
    return word

def extract_drugs_from_prescription(uploaded_prescription):
    """Simple demo logic to extract capitalized potential drug names from image."""
    try:
        reader = load_ocr_model()
        image = Image.open(uploaded_prescription).convert('RGB')
        image_np = np.array(image)
        
        # detail=0 returns just the strings instead of bounding boxes + text + confidence
        results = reader.readtext(image_np, detail=0)
        
        extracted_text = " ".join(results)
        
        # Very naive drug extraction for demo: Look for words > 4 chars that are mostly alphabetic
        potential_drugs = []
        for word in extracted_text.split():
            clean_word = "".join(c for c in word if c.isalpha())
            if len(clean_word) > 4:
                potential_drugs.append(clean_word.lower())
                
        # Deduplicate before API calls
        unique_words = list(set(potential_drugs))
        
        # Concurrently spell check the words to correct OCR mistakes
        corrected_drugs = []
        if unique_words:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_word = {executor.submit(get_spelling_suggestion, w): w for w in unique_words}
                for future in concurrent.futures.as_completed(future_to_word):
                    corrected_word = future.result()
                    if corrected_word:
                        corrected_drugs.append(corrected_word.capitalize())
                
        # Remove duplicates again (in case multiple misspellings map to same drug)
        return list(set(corrected_drugs))
        
    except Exception as e:
        st.error(f"Failed to scan prescription: {e}")
        return []

def extract_label_data(image_file):
    reader = load_ocr_model()
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)
    results = reader.readtext(image_np)
    
    extracted_text = " ".join([text for _, text, _ in results])
    
    # 1. Look for Expiry Dates
    # Improved regex to catch short months and varying punctuation: 'EXP 12/25', 'EXP OCT 25', 'Expiry: 12-2025'
    exp_pattern = r'(?i)(?:exp(?:iry)?|exd)\.?\s*(?:date)?[\s\:\.]*([0-9]{2}[\/\-\.\s]*[0-9]{2,4}|[A-Za-z]{3}[\/\-\.\s]*[0-9]{2,4})'
    exp_matches = re.findall(exp_pattern, extracted_text)
    expiry_date = exp_matches[0] if exp_matches else None
    
    # 2. Look for Rx / NDC / Batch / Lot
    # Improved regex for batch numbers: 'B.No.', 'B. No', 'Batch:', 'Lot'
    lot_pattern = r'(?i)(?:lot|batch|b\.?no\.?)[\s\:\.]*([A-Za-z0-9\-]+)'
    lot_matches = re.findall(lot_pattern, extracted_text)
    lot_number = lot_matches[0] if lot_matches else None
    
    # 3. Predict Drug Name using Bounding Box Heights & RxNorm Validation
    text_fragments = []
    for bbox, text, conf in results:
        top_left = bbox[0]
        bottom_left = bbox[3]
        height = abs(bottom_left[1] - top_left[1])
        
        # Only consider words with letters, > 3 chars
        clean_text = "".join(c for c in text if c.isalpha())
        if len(clean_text) > 3:
            text_fragments.append((clean_text.capitalize(), height, conf))
            
    # Sort by height descending
    text_fragments.sort(key=lambda x: x[1], reverse=True)
    
    potential_names = []
    # Take the top 10 tallest words and validate them against RxNorm
    top_candidates = [frag for frag, height, conf in text_fragments[:10]]
    
    def validate_drug(word):
        import difflib
        corrected = get_spelling_suggestion(word.lower())
        name_to_check = corrected.capitalize() if corrected else word
        
        # Check if the OCR word is at least 70% similar to the spell-checked valid word
        similarity = difflib.SequenceMatcher(None, word.lower(), name_to_check.lower()).ratio()
        
        if similarity >= 0.70:
            rxcui = get_rxcui(name_to_check)
            if rxcui:
                return name_to_check
        return None

    if top_candidates:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_word = {executor.submit(validate_drug, w): w for w in top_candidates}
            for future in concurrent.futures.as_completed(future_to_word):
                valid_name = future.result()
                if valid_name and valid_name not in potential_names:
                    potential_names.append(valid_name)

    # Score logic
    validity_score = 0
    reasons = []
    
    if expiry_date:
        validity_score += 40
        reasons.append(f"✅ Found Expiry Date: {expiry_date}")
    else:
        reasons.append("❌ Missing Expiry Date")
        
    if lot_number:
        validity_score += 30
        reasons.append(f"✅ Found Batch/Lot No: {lot_number}")
    else:
        reasons.append("❌ Missing Batch/Lot No")
        
    if len(potential_names) > 0:
        validity_score += 30
        reasons.append(f"✅ Detected potential drug names (Largest Text).")
    else:
        reasons.append("❌ No clear drug names detected.")
        
    status = "Verified / Legal" if validity_score >= 70 else "Suspicious / Incomplete Label"
    
    return {
        "text": extracted_text,
        "expiry": expiry_date,
        "lot": lot_number,
        "names": list(set(potential_names))[:5],
        "score": validity_score,
        "status": status,
        "reasons": reasons
    }

# ---------------------------------------------------------
# ---------------------------------------------------------
# UI Layout & Custom CSS
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Professional styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .medicine-card {
        background-color: var(--background-color);
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border-left: 4px solid #17a2b8;
        border: 1px solid var(--secondary-background-color);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ MedVigilant Core")
st.markdown("Your smart medical safety assistant for **drug-drug interaction alerts** and **skin rash triage**.")

# ----------------- SIDEBAR: Virtual Medicine Cabinet -----------------
with st.sidebar:
    st.write(f"👤 Logged in as: **{st.session_state.username}**")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        cookie_manager.delete("medvigilant_auth")
        st.rerun()
    st.divider()

    st.header("💊 Virtual Medicine Cabinet")
    
    # Module A: Interaction Guard
    new_drug = st.text_input("Manually enter a drug name:")
    drug_image_file = st.file_uploader("Optional: Upload picture of this medicine", type=["jpg", "jpeg", "png"], key="drug_img")
    
    if st.button("Add Drug", type="primary"):
        if new_drug:
            with st.spinner(f"Checking '{new_drug}'..."):
                rxcui = get_rxcui(new_drug)
                
                if rxcui:
                    existing_drug = cabinet_col.find_one({"rxcui": rxcui, "username": st.session_state.username})
                    if existing_drug:
                        st.warning(f"⚠️ **{new_drug}** is already in your cabinet!")
                    else:
                        is_dangerous, conflict_msg = check_interactions(new_drug)
                        
                        if is_dangerous:
                            st.error(f"🚨 **DANGER!** {conflict_msg}")
                        else:
                            image_url = None
                            if drug_image_file:
                                try:
                                    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
                                    Image.open(drug_image_file).convert('RGB').save(temp_filename)
                                    upload_result = cloudinary.uploader.upload(temp_filename, folder="MedVigilant")
                                    os.remove(temp_filename)
                                    image_url = upload_result.get("secure_url")
                                except Exception as e:
                                    st.error(f"Failed to upload image: {e}")
                                    
                            cabinet_col.insert_one({"name": new_drug, "rxcui": rxcui, "username": st.session_state.username, "image_url": image_url})
                            st.success(f"✅ **{new_drug}** added safely!")
                else:
                    st.warning(f"⚠️ Could not find chemical ID for **{new_drug}**. Try another name.")
        else:
            st.warning("Please enter a drug name.")
            
    st.divider()
    st.divider()
    
    with st.expander("📝 Prescription Scanner", expanded=False):
        presc_file = st.file_uploader("Upload a prescription to extract drugs", type=["jpg", "jpeg", "png"], key="presc")
        
        # Store extracted drugs in session state so multiselect widget doesn't reset on stream-rerun
        if "extracted_drugs" not in st.session_state:
            st.session_state.extracted_drugs = []
            
        if presc_file:
            if st.button("Scan Prescription"):
                with st.spinner("Scanning prescription..."):
                    detected_drugs = extract_drugs_from_prescription(presc_file)
                    if detected_drugs:
                        st.session_state.extracted_drugs = detected_drugs
                        st.success("Extraction complete!")
                    else:
                        st.info("No text readable as a drug was found.")
        
        if st.session_state.extracted_drugs:
            selected_drugs = st.multiselect(
                "Select valid drugs to add to your cabinet:",
                options=st.session_state.extracted_drugs,
                default=st.session_state.extracted_drugs
            )
            
            if st.button("Add Selected to Cabinet", type="primary", use_container_width=True):
                success_count = 0
                for d in selected_drugs:
                    rxcui = get_rxcui(d)
                    if rxcui:
                        existing_drug = cabinet_col.find_one({"rxcui": rxcui, "username": st.session_state.username})
                        if existing_drug:
                            st.warning(f"⚠️ **{d}** is already in your cabinet!")
                        else:
                            is_dangerous, conflict_msg = check_interactions(d)
                            if is_dangerous:
                                st.error(f"🚨 **DANGER!** Interaction involving **{d}**: {conflict_msg}")
                            else:
                                cabinet_col.insert_one({"name": d, "rxcui": rxcui, "username": st.session_state.username})
                                success_count += 1
                                st.success(f"✅ **{d}** added safely!")
                    else:
                        st.warning(f"⚠️ Could not find chemical ID for **{d}**.")
                        
                if success_count > 0:
                    st.session_state.extracted_drugs = [] # Clear the scanner list after successful add

            if presc_file and st.button("Save Prescription to History", use_container_width=True):
                with st.spinner("Uploading and saving to history..."):
                    try:
                        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
                        Image.open(presc_file).convert('RGB').save(temp_filename)
                        upload_result = cloudinary.uploader.upload(temp_filename, folder="MedVigilant")
                        os.remove(temp_filename)
                        image_url = upload_result.get("secure_url")
                        
                        prescriptions_col.insert_one({
                            "username": st.session_state.username,
                            "image_url": image_url,
                            "extracted_drugs": st.session_state.extracted_drugs,
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.success("Prescription saved successfully!")
                    except Exception as e:
                        st.error(f"Failed to save prescription: {e}")

    with st.expander("📜 Your Saved Prescriptions", expanded=False):
        saved_presc = list(prescriptions_col.find({"username": st.session_state.username}).sort("_id", -1))
        if not saved_presc:
            st.info("No saved prescriptions.")
        else:
            for item in saved_presc:
                st.markdown(f"**Date:** {item.get('date', 'Unknown')}")
                st.image(item.get('image_url'), use_container_width=True)
                st.write(f"**Drugs:** {', '.join(item.get('extracted_drugs', []))}")
                st.write("---")

    st.divider()
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Your Cabinet")
    with col2:
        if st.button("Clear All", use_container_width=True):
            cabinet_col.delete_many({"username": st.session_state.username})
            st.rerun()

    current_drugs = list(cabinet_col.find({"username": st.session_state.username}))
    if not current_drugs:
        st.info("Your cabinet is empty.")
    else:
        # Scrollable container for better UI when many medicines are added
        with st.container(height=400):
            for item in current_drugs:
                img_tag = f'<br/><img src="{item.get("image_url")}" style="width:100%; max-height:100px; object-fit:cover; border-radius:4px; margin-top:5px;"/>' if item.get("image_url") else ""
                st.markdown(f"""
                <div class="medicine-card">
                    <span style="font-weight: 600; font-size: 1.1em;">💊 {item.get('name')}</span><br/>
                    <span style="font-size: 0.85em; opacity: 0.7;">RxNorm ID: {item.get('rxcui')}</span>
                    {img_tag}
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Remove ❌", key=f"del_{item['_id']}", use_container_width=True):
                    cabinet_col.delete_one({"_id": item["_id"]})
                    st.rerun()
                st.write("") # small spacing

# ----------------- MAIN: Medicine Label Scanner -----------------
st.header("🏷️ Medicine Label Scanner")
st.markdown("Scan a medicine box or bottle to extract manufacturing details and predict validity/legality.")

label_file = st.file_uploader("Upload an image of a medicine label", type=["jpg", "jpeg", "png"], key="label_scan_input")

# Store extracted drugs in session state so multiselect widget doesn't reset on stream-rerun
if "label_extracted_drugs" not in st.session_state:
    st.session_state.label_extracted_drugs = []

if label_file:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.image(label_file, caption="Uploaded Label", use_container_width=True)
        
    with col_b:
        if st.button("Run OCR & Validity Engine", key="run_label_ocr"):
            with st.spinner("Analyzing text and validating medicines against RxNorm..."):
                label_data = extract_label_data(label_file)
                
                # Store valid names for the multiselect
                if label_data['names']:
                    st.session_state.label_extracted_drugs = label_data['names']
                else:
                    st.session_state.label_extracted_drugs = []
                
                if label_data['status'] == "Verified / Legal":
                    st.success(f"**Status:** {label_data['status']}")
                else:
                    st.error(f"**Status:** {label_data['status']}")
                    
                st.progress(label_data['score'] / 100.0)
                st.markdown(f"**Validity Score:** {label_data['score']}/100")
                
                st.write("**Extracted Details:**")
                st.write(f"- **Expiry Date:** {label_data['expiry'] or '*Not Found*'}")
                st.write(f"- **Batch/Lot:** {label_data['lot'] or '*Not Found*'}")
                
                with st.expander("Show Logic Breakdown"):
                    for r in label_data['reasons']:
                        st.write(f"- {r}")

if st.session_state.label_extracted_drugs:
    st.markdown("### 💊 Detected Medicines")
    st.info("The following medicines were confidently detected and verified against the RxNorm database.")
    
    selected_label_drugs = st.multiselect(
        "Select valid medicines to add to your cabinet:",
        options=st.session_state.label_extracted_drugs,
        default=st.session_state.label_extracted_drugs,
        key="label_multiselect"
    )
    
    if st.button("Add Selected to Cabinet", type="primary", key="add_label_drugs"):
        success_count = 0
        for d in selected_label_drugs:
            rxcui = get_rxcui(d)
            if rxcui:
                existing_drug = cabinet_col.find_one({"rxcui": rxcui, "username": st.session_state.username})
                if existing_drug:
                    st.warning(f"⚠️ **{d}** is already in your cabinet!")
                else:
                    is_dangerous, conflict_msg = check_interactions(d)
                    if is_dangerous:
                        st.error(f"🚨 **DANGER!** Interaction involving **{d}**: {conflict_msg}")
                    else:
                        cabinet_col.insert_one({"name": d, "rxcui": rxcui, "username": st.session_state.username, "image_url": None})
                        success_count += 1
                        st.success(f"✅ **{d}** added safely!")
            else:
                st.warning(f"⚠️ Could not verify **{d}** with RxNorm.")
                
        if success_count > 0:
            st.session_state.label_extracted_drugs = []

st.divider()

@st.cache_resource
def load_disease_model():
    from transformers import pipeline
    # Load authentic DINOv2 31-class skin disease classifier
    return pipeline("image-classification", model="Jayanth2002/dinov2-base-finetuned-SkinDisease")

# ----------------- MAIN: DermAlert (Skin AI) -----------------
st.header("🩺 DermAlert: Skin AI Triage")
st.markdown("Upload a photo of a skin condition to receive a preliminary triage assessment.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        st.subheader("Analysis")
        @st.cache_data(show_spinner=False)
        def get_disease_description(disease_name):
            try:
                # Clean up specific class names for better Wiki matching
                search_term = disease_name.replace("_s", "'s").split(" / ")[0]
                url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exsentences=3&exlimit=1&titles={search_term}&explaintext=1&format=json"
                headers = {'User-Agent': 'MedVigilant/1.0 (test@example.com)'}
                res = requests.get(url, headers=headers, timeout=3).json()
                pages = res.get('query', {}).get('pages', {})
                extract = list(pages.values())[0].get('extract', '')
                if extract and not extract.endswith(":"):
                    return extract
            except Exception:
                pass
            return f"{disease_name.title()} is a clinical condition. Please consult your physician for a proper diagnosis."

        # Module B logic: Authentic Transformers Pipeline
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            with st.spinner("Analyzing image via DINOv2 Deep Learning model..."):
                classifier = load_disease_model()
                # Run inference
                results = classifier(image)
                
            # the pipeline returns a list of dicts: [{'label': 'Melanoma', 'score': 0.95}, ...]
            st.subheader("Top Predictions")

            for i, pred in enumerate(results[:3]):
                label = pred['label']
                score = pred['score'] * 100
    
                st.write(f"{i+1}. **{label}** — {score:.2f}%")
                st.progress(score / 100)
            
            # Risk Mapping for the 31 DINOv2 classes
            red_urgent = [
                "Basal Cell Carcinoma", "Melanoma", "squamous cell carcinoma", 
                "actinic keratosis", "Leprosy Borderline", "Leprosy Lepromatous", 
                "Leprosy Tuberculoid", "Mycosis Fungoides"
            ]
            yellow_consult = [
                "Herpes Simplex", "Impetigo", "Lupus Erythematosus Chronicus Discoides", 
                "Psoriasis", "Tinea Corporis", "Tinea Nigra", "Tungiasis", "Darier_s Disease", 
                "Epidermolysis Bullosa Pruriginosa", "Hailey-Hailey Disease", "Lichen Planus", 
                "Neurofibromatosis", "Papilomatosis Confluentes And Reticulate", "Pityriasis Rosea", 
                "Porokeratosis Actinic", "Larva Migrans"
            ]
            green_common = [
                "Molluscum Contagiosum", "Pediculosis Capitis", "dermatofibroma", 
                "nevus", "pigmented benign keratosis", "seborrheic keratosis", "vascular lesion"
            ]
            
            if disease in red_urgent:
                level_str = "Red (Urgent)"
                st.error(f"🔴 **Risk Level: {level_str}**\n\nHigh-risk condition detected. Seek medical attention immediately.")
            elif disease in yellow_consult:
                level_str = "Yellow (Consult GP)"
                st.warning(f"🟡 **Risk Level: {level_str}**\n\nSigns of potential clinical condition. Please schedule a consultation with your GP.")
            else:
                level_str = "Green (Common/Benign)"
                st.success(f"🟢 **Risk Level: {level_str}**\n\nLikely benign or very common condition detected. Monitor for changes.")
                
            st.divider()
            st.markdown(f"### 🔬 Prediction: **{disease.title()}**")
            st.progress(prob / 100.0)
            st.caption(f"Model Confidence: **{prob:.1f}%** (DINOv2 Vision Transformer)")
            
            with st.expander("📖 Medical Overview", expanded=True):
                wiki_summary = get_disease_description(disease)
                st.write(wiki_summary)
                st.info("⚠️ This result is predicted by an AI model and is **not an official medical diagnosis**. Always verify with a healthcare professional.")
            
            st.divider()
            st.subheader("💊 Medicine Tracker Correlation")
            cabinet_drugs = list(cabinet_col.find({"username": st.session_state.username}))
            found_culprit = False
            
            if cabinet_drugs:
                with st.spinner("Cross-referencing your medicines with FDA Adverse Event reports..."):
                    culprits = []
                    # Thread pool for faster concurrent checking
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_drug = {executor.submit(check_drug_rash_link, d['name']): d['name'] for d in cabinet_drugs}
                        for future in concurrent.futures.as_completed(future_to_drug):
                            is_linked = future.result()
                            if is_linked:
                                culprits.append(future_to_drug[future])
                                
                    if culprits:
                        found_culprit = True
                        for c in culprits:
                            st.warning(f"⚠️ **{c}** in your cabinet is known to cause skin issues like rashes or erythema (Source: OpenFDA). Please consult your doctor regarding this medication.")
            
            if not found_culprit:
                st.info("If you are unsure of the cause of this skin condition, please consult a medical professional.")
                
            # Upload to Cloudinary & Save to MongoDB
            with st.spinner("Uploading to Cloudinary and saving to history..."):
                temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
                image.save(temp_filename)
                
                upload_result = cloudinary.uploader.upload(temp_filename, folder="MedVigilant")
                os.remove(temp_filename)
                
                image_url = upload_result.get("secure_url")
                
                # Save to MongoDB
                triage_record = {
                    "username": st.session_state.username,
                    "risk_level": level_str,
                    "confidence": f"{prob:.2f}% (DINOv2)",
                    "image_url": image_url
                }
                triage_col.insert_one(triage_record)
                
                st.info("✅ Saved to triage history.")
                st.markdown(f"[View Uploaded Image]({image_url})")

        except Exception as e:
            st.error(f"Error processing image: {e}")
