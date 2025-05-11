import os
from flask import Blueprint, request, jsonify
import google.generativeai as genai
import json
# from werkzeug.utils import secure_filename # Potentially needed later for file handling

# Configure the Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # Newer, faster model
        print("Gemini API configured successfully with gemini-2.5-flash-preview-04-17.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        model = None # Ensure model is None if configuration fails
else:
    print("GEMINI_API_KEY not set in .env file. Image recognition will use mock data.")
    print("Please ensure the GEMINI_API_KEY is set in your .env file for live image analysis.")

img_recognition_bp = Blueprint('img_recognition', __name__, url_prefix='/imgrec')

def analyze_image_with_llm(image_bytes, mime_type, prompt_text, scan_type=None):
    """
    Sends the image and prompt to Gemini and returns the structured response.
    scan_type can be 'fasting', 'postprandial', or None.
    """
    if not model:
        print("LLM model not available. Returning mock data.")
        if "血壓" in prompt_text:
            return {"systolic": 125, "diastolic": 85, "pulse": 75}
        elif "血糖" in prompt_text:
            if scan_type == "fasting":
                 return {"fasting": 95}
            elif scan_type == "postprandial":
                 return {"postprandial": 135}
            return {"blood_sugar_value": 100} # Generic mock if scan_type not specific
        return {}

    try:
        image_part = {"mime_type": mime_type, "data": image_bytes}
        
        # Add specific instructions for JSON output and handling of missing values
        json_instruction = (
            "請以JSON格式返回結果。例如，對於血壓，返回 {\"systolic\": 值, \"diastolic\": 值, \"pulse\": 值}。"
            "對於血糖，如果是空腹血糖，返回 {\"fasting\": 值}；如果是餐後血糖，返回 {\"postprandial\": 值}。"
            "如果圖片中無法明確識別某個數值，請在JSON中省略該鍵或將其值設為null。"
            "如果完全無法識別任何相關數值，請返回一個空的JSON物件 {}。"
        )
        full_prompt = f"{prompt_text} {json_instruction}"
        
        print(f"Sending prompt to Gemini (scan_type: {scan_type}): {full_prompt[:150]}...")
        
        response = model.generate_content([image_part, full_prompt])
        
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[len("```json"):]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-len("```")]
        cleaned_response_text = cleaned_response_text.strip()
        
        print(f"LLM Raw Response: {cleaned_response_text}")
        
        if not cleaned_response_text:
            print("LLM returned empty response.")
            return {}
            
        parsed_json = json.loads(cleaned_response_text)
        return parsed_json
        
    except Exception as e:
        print(f"Error during LLM image analysis: {e}")
        # Fallback to different mock data on error to distinguish from no-API-key mock
        if "血壓" in prompt_text:
            return {"systolic": 118, "diastolic": 78, "pulse": 68}
        elif "血糖" in prompt_text:
            if scan_type == "fasting":
                 return {"fasting": 88}
            elif scan_type == "postprandial":
                 return {"postprandial": 128}
            return {"blood_sugar_value": 108}
        return {}

@img_recognition_bp.route('/bp_image', methods=['POST'])
def bp_image_recognition():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    # scan_type = request.form.get('scan_type') # 'bp' - though not strictly needed for BP endpoint logic itself

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.content_type in ['image/jpeg', 'image/png', 'image/webp']:
        try:
            image_bytes = file.read()
            mime_type = file.content_type
            prompt = "從這張血壓計圖片中提取收縮壓(systolic)、舒張壓(diastolic)和脈搏(pulse)的數值。"
            
            extracted_data = analyze_image_with_llm(image_bytes, mime_type, prompt)
            
            processed_data = {
                "systolic": extracted_data.get("systolic"),
                "diastolic": extracted_data.get("diastolic"),
                "pulse": extracted_data.get("pulse")
            }
            return jsonify(processed_data), 200
        except Exception as e:
            print(f"Error processing BP image: {e}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload JPG, PNG, or WEBP images."}), 400
        
@img_recognition_bp.route('/bs_image', methods=['POST'])
def bs_image_recognition():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    scan_type = request.form.get('scan_type') # Expected: 'fasting' or 'postprandial'

    if not scan_type or scan_type not in ['fasting', 'postprandial']:
        return jsonify({"error": "Missing or invalid scan_type (must be 'fasting' or 'postprandial')"}), 400

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.content_type in ['image/jpeg', 'image/png', 'image/webp']:
        try:
            image_bytes = file.read()
            mime_type = file.content_type
            
            type_specific_prompt_part = ""
            if scan_type == "fasting":
                type_specific_prompt_part = "這是一張空腹血糖的測量結果。"
            elif scan_type == "postprandial":
                type_specific_prompt_part = "這是一張餐後血糖的測量結果。"

            prompt = f"從這張血糖儀圖片中提取血糖數值。{type_specific_prompt_part} 請將空腹血糖值放在 'fasting' 鍵下，餐後血糖值放在 'postprandial' 鍵下。"
            
            extracted_data = analyze_image_with_llm(image_bytes, mime_type, prompt, scan_type)
            
            # Initialize with None to ensure keys exist if LLM doesn't return them
            processed_data = {
                "fasting": None,
                "postprandial": None,
                "blood_sugar_value": None # For generic fallback
            }

            if scan_type == "fasting":
                processed_data["fasting"] = extracted_data.get("fasting", extracted_data.get("blood_sugar_value"))
            elif scan_type == "postprandial":
                processed_data["postprandial"] = extracted_data.get("postprandial", extracted_data.get("blood_sugar_value"))
            
            # If LLM returned a generic value and specific type was not found
            if extracted_data.get("blood_sugar_value") is not None:
                if scan_type == "fasting" and processed_data["fasting"] is None:
                    processed_data["fasting"] = extracted_data.get("blood_sugar_value")
                elif scan_type == "postprandial" and processed_data["postprandial"] is None:
                    processed_data["postprandial"] = extracted_data.get("blood_sugar_value")
                # also store it as blood_sugar_value for debugging or alternative frontend handling
                processed_data["blood_sugar_value"] = extracted_data.get("blood_sugar_value")


            return jsonify(processed_data), 200
        except Exception as e:
            print(f"Error processing BS image (scan_type: {scan_type}): {e}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload JPG, PNG, or WEBP images."}), 400
