"""
Runtime Safety Guardrails for MedVigilant
==========================================

Provides input sanitization, output validation, image verification,
and confidence gating for all AI-facing pipelines.

Each function returns a tuple of (sanitized_value, list_of_flags).
Empty flags = safe input/output. Non-empty flags = suspicious content detected.

Usage:
    from safety.guardrails import sanitize_input, validate_output
    
    safe_text, flags = sanitize_input(user_text)
    if flags:
        handle_flagged_input(flags)
"""

import re
import unicodedata
from PIL import Image
import io


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_TEXT_LENGTH = 200         # Max chars for drug name inputs
MAX_QUERY_LENGTH = 500        # Max chars for LLM query inputs
MAX_IMAGE_DIMENSION = 5000    # Max pixel width/height
MIN_IMAGE_DIMENSION = 10      # Min pixel width/height
MAX_FILE_SIZE_MB = 10         # Max file size in MB

# Prompt injection patterns — common adversarial attack signatures
INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all|prior)\s+(instructions|rules|prompts)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a|an)\s+",
    r"(system|internal)\s*prompt",
    r"forget\s+(everything|your\s+rules|all\s+instructions)",
    r"repeat\s+(your|the)\s+(system\s+)?prompt",
    r"what\s+are\s+your\s+(instructions|rules)",
    r"disregard\s+(all|any|previous)",
    r"override\s+(safety|previous|all)",
    r"jailbreak",
    r"DAN\s+mode",
]

# NoSQL / MongoDB injection patterns
NOSQL_PATTERNS = [
    r"\$gt\b",
    r"\$ne\b",
    r"\$regex\b",
    r"\$where\b",
    r"\$or\b",
    r"\$and\b",
    r"\$exists\b",
    r"\$nin\b",
]

# SQL injection patterns
SQL_PATTERNS = [
    r";\s*DROP\s+",
    r";\s*DELETE\s+",
    r";\s*INSERT\s+",
    r";\s*UPDATE\s+",
    r"'\s*OR\s+'1'\s*=\s*'1",
    r"--\s*$",
    r"UNION\s+SELECT",
]

# Dangerous medical content patterns
UNSAFE_MEDICAL_PATTERNS = [
    r"(take|consume|ingest)\s+(all|everything)\s+at\s+once",
    r"(lethal|fatal|deadly)\s+(dose|amount|quantity)",
    r"(cause|inflict)\s+harm",
    r"how\s+to\s+(poison|overdose|kill)",
    r"(suicide|self[- ]harm)",
    r"(\d{3,})\s*mg",  # Suspiciously high dosages (3+ digits)
]

# Output validation patterns — things the LLM should NEVER say
UNSAFE_OUTPUT_PATTERNS = [
    r"(take|consume)\s+\d+\s*(mg|ml|tablets?|pills?|capsules?)",  # Specific dosage
    r"you\s+(have|definitely\s+have|are\s+suffering\s+from)\s+",   # Diagnosis
    r"(stop\s+taking|discontinue)\s+your\s+(medication|medicine)", # Dangerous advice
    r"(double|triple|increase)\s+(the|your)\s+dose",              # Dosage changes
    r"(no\s+need|don't\s+need)\s+to\s+(see|visit|consult)\s+(a\s+)?(doctor|physician)",
]

# Cyrillic characters that look like Latin (homoglyph attacks)
CYRILLIC_HOMOGLYPHS = {
    '\u0410': 'A', '\u0412': 'B', '\u0421': 'C', '\u0415': 'E',
    '\u041d': 'H', '\u041a': 'K', '\u041c': 'M', '\u041e': 'O',
    '\u0420': 'P', '\u0422': 'T', '\u0425': 'X', '\u0423': 'Y',
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0443': 'y', '\u0445': 'x',
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. TEXT INPUT SANITIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sanitize_input(text: str, max_length: int = MAX_QUERY_LENGTH) -> tuple:
    """
    Validates and sanitizes user text input (drug names, LLM queries).
    
    Checks for:
        - Prompt injection patterns
        - SQL / NoSQL injection attempts
        - Unsafe medical content
        - Excessive length
        - Unicode homoglyph attacks (e.g., Cyrillic letters mimicking Latin)
        - XSS / template injection
    
    Args:
        text: Raw user input string
        max_length: Maximum allowed character count
        
    Returns:
        tuple: (sanitized_text: str, flags: list[str])
               flags is empty if input is safe
    """
    if not text or not isinstance(text, str):
        return "", ["empty_input"]
    
    flags = []
    sanitized = text.strip()
    
    # ── Length check ──
    if len(sanitized) > max_length:
        flags.append(f"excessive_length ({len(sanitized)} chars, max {max_length})")
        sanitized = sanitized[:max_length]
    
    # ── Homoglyph detection ──
    homoglyphs_found = []
    for char in sanitized:
        if char in CYRILLIC_HOMOGLYPHS:
            homoglyphs_found.append(f"'{char}' → '{CYRILLIC_HOMOGLYPHS[char]}'")
    if homoglyphs_found:
        flags.append(f"homoglyph_attack (found: {', '.join(homoglyphs_found[:3])})")
        # Replace Cyrillic with Latin equivalents
        for cyrillic, latin in CYRILLIC_HOMOGLYPHS.items():
            sanitized = sanitized.replace(cyrillic, latin)
    
    # ── Prompt injection check ──
    text_lower = sanitized.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            flags.append(f"prompt_injection ({pattern[:40]}...)")
            break  # One flag is enough
    
    # ── NoSQL injection check ──
    for pattern in NOSQL_PATTERNS:
        if re.search(pattern, sanitized):
            flags.append(f"nosql_injection ({pattern})")
            break
    
    # ── SQL injection check ──
    for pattern in SQL_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            flags.append(f"sql_injection ({pattern[:30]}...)")
            break
    
    # ── XSS / Template injection ──
    if re.search(r"<script|<img\s|javascript:|{{.*}}|\$\{.*\}", sanitized, re.IGNORECASE):
        flags.append("xss_or_template_injection")
        sanitized = re.sub(r"[<>{}]", "", sanitized)
    
    # ── Unsafe medical content ──
    for pattern in UNSAFE_MEDICAL_PATTERNS:
        if re.search(pattern, text_lower):
            flags.append(f"unsafe_medical_content ({pattern[:40]}...)")
            break
    
    # ── Non-printable characters ──
    cleaned = "".join(
        ch for ch in sanitized 
        if unicodedata.category(ch)[0] != 'C'  # Remove control characters
    )
    if cleaned != sanitized:
        flags.append("non_printable_characters_removed")
        sanitized = cleaned
    
    return sanitized, flags


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. IMAGE INPUT VALIDATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_image(file_data) -> tuple:
    """
    Validates uploaded image files for safety before passing to AI models.
    
    Checks for:
        - Valid image format (actually openable by PIL)
        - Reasonable dimensions (not degenerate 1x1 or enormous)
        - File size within limits
        - Unusual pixel distributions (potential adversarial patches)
    
    Args:
        file_data: Streamlit UploadedFile object or file-like object
        
    Returns:
        tuple: (is_valid: bool, flags: list[str])
    """
    flags = []
    
    if file_data is None:
        return False, ["no_file_provided"]
    
    # ── File size check ──
    try:
        file_data.seek(0, 2)  # Seek to end
        file_size_mb = file_data.tell() / (1024 * 1024)
        file_data.seek(0)     # Reset to beginning
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            flags.append(f"file_too_large ({file_size_mb:.1f}MB, max {MAX_FILE_SIZE_MB}MB)")
            return False, flags
        
        if file_size_mb < 0.001:  # Less than 1KB
            flags.append("file_suspiciously_small")
    except Exception:
        flags.append("unable_to_read_file_size")
    
    # ── Actual image validation ──
    try:
        file_data.seek(0)
        img = Image.open(file_data)
        img.verify()  # Verify it's a real image, not a renamed .exe
        
        # Re-open after verify (verify() can only be called once)
        file_data.seek(0)
        img = Image.open(file_data)
        width, height = img.size
        
        # ── Dimension checks ──
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            flags.append(f"degenerate_dimensions ({width}x{height})")
        
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            flags.append(f"excessive_dimensions ({width}x{height})")
        
        # ── Aspect ratio check (extremely skewed images are suspicious) ──
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > 20:
            flags.append(f"suspicious_aspect_ratio ({aspect:.1f}:1)")
        
        # ── Pixel distribution check (adversarial patch detection) ──
        try:
            import numpy as np
            img_array = np.array(img.convert('RGB'))
            
            # Check if image is nearly uniform (blank/solid color)
            std_dev = np.std(img_array)
            if std_dev < 5.0:
                flags.append(f"near_uniform_image (std_dev={std_dev:.1f})")
            
            # Check for extreme pixel values concentration (adversarial noise signature)
            pixel_mean = np.mean(img_array)
            if pixel_mean < 5.0 or pixel_mean > 250.0:
                flags.append(f"extreme_pixel_distribution (mean={pixel_mean:.1f})")
                
        except ImportError:
            pass  # numpy not available, skip pixel analysis
            
        file_data.seek(0)  # Reset for downstream use
        
    except Exception as e:
        flags.append(f"invalid_image_file ({type(e).__name__}: {str(e)[:50]})")
        return False, flags
    
    is_valid = len(flags) == 0
    return is_valid, flags


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. LLM OUTPUT VALIDATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_output(response: str) -> tuple:
    """
    Validates LLM-generated responses before showing to the user.
    
    Checks for:
        - Specific dosage recommendations (model should NEVER give doses)
        - Diagnostic claims ("you have X disease")
        - Dangerous medical advice ("stop taking your medication")
        - System prompt leaks
        - Hallucinated content indicators
    
    Args:
        response: Raw LLM response string
        
    Returns:
        tuple: (safe_response: str, flags: list[str])
    """
    if not response or not isinstance(response, str):
        return "I'm unable to provide a response at this time.", ["empty_response"]
    
    flags = []
    FALLBACK = ("I'm unable to provide a safe response for this query. "
                "Please consult a healthcare professional.")
    
    # ── System prompt leak detection ──
    prompt_leak_indicators = [
        "you are medvigilant",
        "system prompt",
        "my instructions are",
        "i was programmed to",
        "my rules are",
        "here are my instructions",
        "i am designed to",
    ]
    response_lower = response.lower()
    for indicator in prompt_leak_indicators:
        if indicator in response_lower:
            flags.append(f"possible_prompt_leak ({indicator})")
            return FALLBACK, flags
    
    # ── Unsafe output patterns ──
    for pattern in UNSAFE_OUTPUT_PATTERNS:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            flags.append(f"unsafe_output ({match.group()[:50]})")
    
    # ── Check for excessive confidence in medical claims ──
    overconfident_patterns = [
        r"(definitely|certainly|100%)\s+(is|have|caused\s+by)",
        r"i\s+(guarantee|promise|assure)",
        r"there\s+is\s+no\s+(risk|danger|chance)",
    ]
    for pattern in overconfident_patterns:
        if re.search(pattern, response_lower):
            flags.append(f"overconfident_medical_claim ({pattern[:30]})")
    
    if flags:
        return FALLBACK, flags
    
    return response, flags


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. CONFIDENCE GATE (DermAlert)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def confidence_gate(predictions: list, threshold: float = 0.40) -> tuple:
    """
    Gates DermAlert predictions by confidence level.
    
    Prevents low-confidence or ambiguous classifications from being
    presented as definitive diagnoses to the user.
    
    Also detects near-uniform probability distributions, which indicate
    the model is uncertain (likely out-of-distribution input).
    
    Args:
        predictions: List of dicts from HuggingFace pipeline 
                     [{"label": "...", "score": 0.x}, ...]
        threshold: Minimum confidence for the top prediction
        
    Returns:
        tuple: (top_prediction: dict | None, flags: list[str])
               top_prediction is None if confidence is too low
    """
    if not predictions:
        return None, ["no_predictions"]
    
    flags = []
    top = predictions[0]
    top_score = top.get("score", 0)
    
    # ── Low confidence check ──
    if top_score < threshold:
        flags.append(
            f"low_confidence (top={top['label']}: {top_score:.2%}, "
            f"threshold={threshold:.0%})"
        )
    
    # ── Near-uniform distribution check (OOD detection) ──
    if len(predictions) >= 3:
        scores = [p["score"] for p in predictions[:5]]
        score_range = max(scores) - min(scores)
        
        if score_range < 0.10:
            flags.append(
                f"near_uniform_distribution (range={score_range:.3f}), "
                f"likely out-of-distribution input"
            )
    
    # ── Top-2 ambiguity check ──
    if len(predictions) >= 2:
        second_score = predictions[1].get("score", 0)
        margin = top_score - second_score
        
        if margin < 0.05 and top_score < 0.60:
            flags.append(
                f"ambiguous_prediction (margin={margin:.3f} between "
                f"'{top['label']}' and '{predictions[1]['label']}')"
            )
    
    if flags:
        return None, flags
    
    return top, flags
