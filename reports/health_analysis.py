import os
import pandas as pd
import google.generativeai as genai
import pdfkit
from jinja2 import Template
from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import shutil # Used for os.makedirs, but clear_user_data_folder is now in app.py

import io
import base64

# 導入 sklearn 相關模組
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA

# Configure matplotlib for non-interactive backend and CJK fonts
matplotlib.use('Agg')
try:
    # Prioritize common CJK fonts. Ensure these are installed on your system.
    plt.rcParams['font.sans-serif'] = ['Noto Sans TC', 'Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False # Prevent minus sign from being a square
except Exception as e:
    print(f"Warning: Could not set preferred CJK fonts, ensure they are installed: {e}")

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-1.5-flash")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found. AI generation features will fail.")

# Configure wkhtmltopdf
WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH")
config = None
if WKHTMLTOPDF_PATH and os.path.exists(WKHTMLTOPDF_PATH):
    config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
else:
    print("Warning: WKHTMLTOPDF_PATH is not set or the wkhtmltopdf executable does not exist. PDF generation will fail.")

# Helper function (moved from lib.py for self-containment)
# Note: This local clear_user_data_folder is only for the __main__ test block.
# The main application uses the one defined in app.py.
def _local_clear_user_data_folder(user_id: str, subfolder: str):
    """Clears and recreates a specific subfolder within a user's static data directory."""
    # This path needs to be consistent with app.py's UPLOAD_FOLDER setting
    folder_path = os.path.join("static", "user_data", user_id, subfolder)
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Cleared folder: {folder_path}")
        except OSError as e:
            print(f"Error clearing folder {folder_path}: {e}")
    os.makedirs(folder_path, exist_ok=True) # Recreate the folder

# Blood Pressure and Blood Sugar Standards (unchanged)
BP_NORMAL_SYSTOLIC_MAX = 119
BP_NORMAL_DIASTOLIC_MAX = 79
BP_ELEVATED_SYSTOLIC_MIN = 120
BP_ELEVATED_SYSTOLIC_MAX = 129
BP_ELEVATED_DIASTOLIC_MAX = 79
BP_HYPERTENSION_S1_SYSTOLIC_MIN = 130
BP_HYPERTENSION_S1_SYSTOLIC_MAX = 139
BP_HYPERTENSION_S1_DIASTOLIC_MIN = 80
BP_HYPERTENSION_S1_DIASTOLIC_MAX = 89
BP_HYPERTENSION_S2_SYSTOLIC_MIN = 140
BP_HYPERTENSION_S2_DIASTOLIC_MIN = 90
BP_CRISIS_SYSTOLIC_MIN = 180
BP_CRISIS_DIASTOLIC_MIN = 120
BP_LOW_SYSTOLIC_MAX = 90
BP_LOW_DIASTOLIC_MAX = 60

BS_NORMAL_FASTING_MIN = 70
BS_NORMAL_FASTING_MAX = 99
BS_PREDIABETES_FASTING_MIN = 100
BS_PREDIABETES_FASTING_MAX = 125
BS_DIABETES_FASTING_MIN = 126
BS_NORMAL_POSTPRANDIAL_MAX = 139
BS_PREDIABETES_POSTPRANDIAL_MIN = 140
BS_PREDIABETES_POSTPRANDIAL_MAX = 199
BS_DIABETES_POSTPRANDIAL_MIN = 200
BS_HYPOGLYCEMIA_MAX = 69

# Prompts (Adjusted for bullet points in advice)
trend_prompt = """
你是一位健康數據分析師，請根據以下提供的健康數據紀錄（已根據用戶選擇的時間區間篩選），分析數據中是否存在任何顯著的異常趨勢（例如，指標持續升高、持續降低、波動過於劇烈、頻繁超出正常範圍等）。
請提供簡短的觀察結果和針對這些趨勢的初步建議。

請輸出格式如下：
- 🟡 指標變化觀察：[描述觀察到的數據模式，例如：過去一週血壓有輕微上升趨勢，尤其是在晚間。]
- 🔴 健康建議：
  - [根據觀察到的趨勢提供第一條具體建議，例如：建議持續監測血壓，並記錄每日活動和飲食，以找出潛在影響因素。]
  - [提供第二條具體建議，例如：考慮調整晚間作息，確保充足睡眠，避免睡前攝入咖啡因或高糖食物。]
  - [提供第三條具體建議，例如：若血壓持續升高或出現不適，請務必諮詢醫療專業人員，切勿自行調整藥物。]
"""

rag_prompt_template = """
你是長照照護助手，請根據你掌握的知識，針對照顧者的問題提供具體建議。
問題：「{question}」
"""

# HTML Template for PDF Report (Updated to use base64 image embedding)
PDF_REPORT_TEMPLATE = """
<html>
<head>
    <meta charset="utf-8">
    <title>{{ report_title }}</title>
    <style>
        body { font-family: 'Noto Sans TC', 'Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', sans-serif; margin: 20px; color: #333; }
        h1, h2, h3 { color: #0056b3; }
        h1 { text-align: center; margin-bottom: 30px; }
        h2 { border-bottom: 2px solid #0056b3; padding-bottom: 5px; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; word-wrap: break-word; }
        th { background-color: #f2f2f2; font-weight: bold; }
        img { max-width: 90%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; padding: 5px; }
        .analysis-section { margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }
        .analysis-section p { white-space: pre-wrap; line-height: 1.6; }
        .analysis-section ul { margin-left: 20px; padding-left: 0; list-style-type: disc; } /* Added for bullet points */
        .analysis-section li { margin-bottom: 5px; } /* Added for bullet points */
        .footer { text-align: center; margin-top: 40px; font-size: 0.8em; color: #777; }
    </style>
</head>
<body>
    <h1>{{ report_title }}</h1>
    <p>分析期間：{{ time_period_label }}</p>

    {% if data_table_html %}
    <h2>數據總覽</h2>
    {{ data_table_html | safe }}
    {% else %}
    <h2>數據總覽</h2>
    <p>此期間無數據可顯示。</p>
    {% endif %}

    {% if trend_plot_html_tag %} {# Changed from trend_plot_abs_path #}
    <h2>趨勢圖</h2>
    {{ trend_plot_html_tag | safe }} {# Embed base64 image tag #}
    {% endif %}

    {% if trend_analysis_text %}
    <h2>趨勢分析與建議 (AI生成)</h2>
    <div class="analysis-section">
        {# Replace newlines with <br> tags, and convert markdown list to HTML list #}
        {{ trend_analysis_text | replace('\\n', '<br>') | markdown_to_html | safe }}
    </div>
    {% endif %}

    {% if pca_plot_abs_path %} {# This is still a file path for wkhtmltopdf local access #}
    <h2>PCA 主成分分析圖</h2>
    <img src="file:///{{ pca_plot_abs_path }}" alt="PCA 分析圖">
    {% endif %}
    
    {% if pca_interpretation_text %}
    <h2>PCA 分析說明</h2>
    <div class="analysis-section">
        <p>{{ pca_interpretation_text | replace('\\n', '<br>') | safe }}</p>
    </div>
    {% endif %}

    <div class="footer">
        <p>HealthLLM 健康報告 - 生成時間: {{ generation_timestamp }}</p>
    </div>
</body>
</html>
"""

# Custom Jinja2 filter for markdown to HTML conversion
def markdown_to_html(text):
    import markdown
    return markdown.markdown(text)

def parse_markdown_table(markdown_text: str) -> pd.DataFrame:
    # This function is not directly used in the current trend analysis flow,
    # but kept for completeness if other parts of the system use it.
    lines = [line.strip() for line in markdown_text.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if not table_lines or len(table_lines) < 2: # Need at least header and separator
        return pd.DataFrame()
    
    headers_line = table_lines[0]
    headers = [h.strip() for h in headers_line.strip("|").split("|")]
    
    data_rows = []
    if len(table_lines) > 2:
        for line in table_lines[2:]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == len(headers):
                data_rows.append(cells)
            else:
                print(f"Debug: parse_markdown_table - Mismatch in cell count for row: {line}. Expected {len(headers)}, got {len(cells)}.")
                continue 
    
    if not data_rows and headers:
        return pd.DataFrame(columns=headers)
    if not data_rows and not headers:
        return pd.DataFrame()
        
    return pd.DataFrame(data_rows, columns=headers)

def generate_html_for_summary(df: pd.DataFrame, title="健康紀錄分析") -> str:
    # This function is not directly used in the current trend analysis flow,
    # but kept for completeness if other parts of the system use it.
    original_html_template = """
    <html><head><meta charset="utf-8"><style>body { font-family: Arial, sans-serif; } table { border-collapse: collapse; width: 100%; } th, td { border: 1px solid black; padding: 8px; text-align: left; } th { background-color: #f2f2f2; } img { max-width: 100%; margin-top: 20px; }</style></head>
    <body><h2>{{ title }}</h2>{{ table_html | safe }}</body></html>
    """
    template = Template(original_html_template)
    table_html_content = df.to_html(index=False, classes="dataframe", border=0) if not df.empty else "<p>無數據可顯示。</p>"
    return template.render(table_html=table_html_content, title=title)

def generate_pdf_from_html_summary(html_content: str, base_output_dir: str, pdf_filename: str) -> str:
    # This function is not directly used in the current trend analysis flow,
    # but kept for completeness if other parts of the system use it.
    if not config:
        print("Error: wkhtmltopdf configuration not set. Cannot generate PDF summary.")
        return None
    pdf_path = os.path.join(base_output_dir, "summary", pdf_filename) # Adjusted path
    abs_pdf_path = os.path.abspath(pdf_path)
    os.makedirs(os.path.dirname(abs_pdf_path), exist_ok=True)
    try:
        pdfkit.from_string(html_content, abs_pdf_path, configuration=config, options={'enable-local-file-access': ''})
        return pdf_path
    except Exception as e:
        print(f"Error generating PDF from HTML summary: {e}")
        return None

def validate_bp_csv(df):
    # These column names must match exactly what's written to CSV by app.py
    required_columns = [
        'Date', 'Morning_Systolic', 'Morning_Diastolic', 'Morning_Pulse',
        'Midday_Systolic', 'Midday_Diastolic', 'Midday_Pulse',
        'Evening_Systolic', 'Evening_Diastolic', 'Evening_Pulse'
    ]
    actual_columns = df.columns.tolist()
    missing_cols = [col for col in required_columns if col not in actual_columns]
    if missing_cols:
        print(f"Debug: validate_bp_csv - Missing BP columns: {missing_cols}")
        return False
    return True

def validate_sugar_csv(df):
    # These column names must match exactly what's written to CSV by app.py
    # Corrected 'Noon_Fasting' to 'Midday_Fasting' and 'Noon_Postprandial' to 'Midday_Postprandial'
    required_columns = [
        'Date', 'Morning_Fasting', 'Morning_Postprandial',
        'Midday_Fasting', 'Midday_Postprandial', # Corrected
        'Evening_Fasting', 'Evening_Postprandial'
    ]
    actual_columns = df.columns.tolist()
    missing_cols = [col for col in required_columns if col not in actual_columns]
    if missing_cols:
        print(f"Debug: validate_sugar_csv - Missing Sugar columns: {missing_cols}")
        return False
    return True

def analyze_blood_pressure(systolic, diastolic, pulse=None):
    # This function is used by the frontend for real-time status, not directly by trend analysis.
    if not (isinstance(systolic, (int, float)) and isinstance(diastolic, (int, float))):
        return "血壓輸入無效", "請輸入有效的數字作為血壓值。", ""
    if pulse is not None and not isinstance(pulse, (int, float)):
        return "脈搏輸入無效", "請輸入有效的數字作為脈搏值，或留空。", ""
    
    s = int(systolic)
    d = int(diastolic)
    status = "未知血壓狀態"
    advice = "請諮詢醫生以獲得專業評估。"
    normal_range_info = f"理想血壓: 收縮壓 < {BP_NORMAL_SYSTOLIC_MAX + 1} mmHg 且 舒張壓 < {BP_NORMAL_DIASTOLIC_MAX + 1} mmHg。"

    if s >= BP_CRISIS_SYSTOLIC_MIN or d >= BP_CRISIS_DIASTOLIC_MIN:
        status = "高血壓危機"
        advice = "您的血壓非常高，這可能表示高血壓危機。請立即尋求醫療協助！"
    elif s >= BP_HYPERTENSION_S2_SYSTOLIC_MIN or d >= BP_HYPERTENSION_S2_DIASTOLIC_MIN:
        status = "第二期高血壓"
        advice = "您的血壓處於第二期高血壓範圍，建議立即諮詢醫生，可能需要藥物治療和生活方式調整。"
    elif (BP_HYPERTENSION_S1_SYSTOLIC_MIN <= s <= BP_HYPERTENSION_S1_SYSTOLIC_MAX) or \
         (BP_HYPERTENSION_S1_DIASTOLIC_MIN <= d <= BP_HYPERTENSION_S1_DIASTOLIC_MAX):
        status = "第一期高血壓"
        advice = "您的血壓處於第一期高血壓範圍，建議諮詢醫生討論生活方式改變，並定期監測。"
    elif (BP_ELEVATED_SYSTOLIC_MIN <= s <= BP_ELEVATED_SYSTOLIC_MAX) and d <= BP_ELEVATED_DIASTOLIC_MAX:
        status = "血壓升高"
        advice = "您的血壓略高於理想範圍，建議開始注意健康生活方式，如健康飲食、規律運動和減輕壓力。"
    elif s <= BP_NORMAL_SYSTOLIC_MAX and d <= BP_NORMAL_DIASTOLIC_MAX:
        if s < BP_LOW_SYSTOLIC_MAX or d < BP_LOW_DIASTOLIC_MAX:
             status = "血壓偏低"
             advice = "您的血壓在正常範圍內但偏低，如果伴有頭暈、乏力等症狀，請諮詢醫生。"
             normal_range_info = f"一般低血壓參考: 收縮壓 < {BP_LOW_SYSTOLIC_MAX} mmHg 或 舒張壓 < {BP_LOW_DIASTOLIC_MAX} mmHg。"
        else:
            status = "正常血壓"
            advice = "您的血壓在理想範圍，請繼續保持健康的生活習慣。"
    elif s < BP_LOW_SYSTOLIC_MAX or d < BP_LOW_DIASTOLIC_MAX:
        status = "低血壓"
        advice = "您的血壓偏低，如果伴有頭暈、乏力等症狀，請諮詢醫生。"
        normal_range_info = f"一般低血壓參考: 收縮壓 < {BP_LOW_SYSTOLIC_MAX} mmHg 或 舒張壓 < {BP_LOW_DIASTOLIC_MAX} mmHg。"
    else:
        if s > BP_NORMAL_SYSTOLIC_MAX or d > BP_NORMAL_DIASTOLIC_MAX:
            status = "血壓偏高 (需進一步分類)"
            advice = "您的血壓數值超出了正常範圍，建議諮詢醫生以獲得詳細評估和分類。"
        else:
            status = "血壓數據組合特殊"
            advice = "您的血壓數據組合較為特殊或不完整，建議諮詢醫生。"

    pulse_info = ""
    if pulse is not None:
        p = int(pulse)
        pulse_info = f"脈搏: {p} 次/分. "
        if not (60 <= p <= 100):
            pulse_info += "脈搏速率不在常規靜息範圍 (60-100 次/分)，建議注意。"
    
    advice = pulse_info + advice if pulse_info else advice
    return status, advice, normal_range_info

def analyze_blood_sugar(value, measurement_type="fasting"):
    # This function is used by the frontend for real-time status, not directly by trend analysis.
    if not isinstance(value, (int, float)):
        return "血糖輸入無效", "請輸入有效的數字作為血糖值。", ""
    val = int(value)
    status = "未知血糖狀態"
    advice = "請諮詢醫生以獲得專業評估。"
    normal_range_info = ""

    if measurement_type.lower() == "fasting":
        normal_range_info = f"理想空腹血糖: {BS_NORMAL_FASTING_MIN}-{BS_NORMAL_FASTING_MAX} mg/dL"
        if val <= BS_HYPOGLYCEMIA_MAX:
            status = "低血糖 (空腹)"
            advice = f"您的空腹血糖 ({val} mg/dL) 偏低，可能為低血糖。若有不適請立即補充糖分並諮詢醫生。"
        elif val <= BS_NORMAL_FASTING_MAX:
            status = "正常空腹血糖"
            advice = f"您的空腹血糖 ({val} mg/dL) 在理想範圍。"
        elif BS_PREDIABETES_FASTING_MIN <= val <= BS_PREDIABETES_FASTING_MAX:
            status = "糖尿病前期 (空腹)"
            advice = f"您的空腹血糖 ({val} mg/dL) 偏高，屬於糖尿病前期。建議改善飲食、增加運動，並定期追蹤血糖。"
        elif val >= BS_DIABETES_FASTING_MIN:
            status = "糖尿病 (空腹)"
            advice = f"您的空腹血糖 ({val} mg/dL) 明顯偏高，可能已達糖尿病標準。請立即諮詢醫生進行進一步檢查和治療。"
        else:
            status = "血糖偏低 (空腹)"
            advice = f"您的空腹血糖 ({val} mg/dL) 略低於常規正常範圍下限，請注意觀察，若有不適請諮詢醫生。"
    elif measurement_type.lower() == "postprandial":
        normal_range_info = f"理想餐後血糖 (餐後2小時): < {BS_NORMAL_POSTPRANDIAL_MAX + 1} mg/dL"
        if val <= BS_HYPOGLYCEMIA_MAX:
            status = "低血糖 (餐後)"
            advice = f"您的餐後血糖 ({val} mg/dL) 偏低，可能為低血糖。若有不適請立即補充糖分並諮詢醫生。"
        elif val <= BS_NORMAL_POSTPRANDIAL_MAX:
            status = "正常餐後血糖"
            advice = f"您的餐後血糖 ({val} mg/dL) 在理想範圍。"
        elif BS_PREDIABETES_POSTPRANDIAL_MIN <= val <= BS_PREDIABETES_POSTPRANDIAL_MAX:
            status = "糖尿病前期 (餐後)"
            advice = f"您的餐後血糖 ({val} mg/dL) 偏高，屬於糖尿病前期。建議改善飲食、增加運動，並定期追蹤血糖。"
        elif val >= BS_DIABETES_POSTPRANDIAL_MIN:
            status = "糖尿病 (餐後)"
            advice = f"您的餐後血糖 ({val} mg/dL) 明顯偏高，可能已達糖尿病標準。請立即諮詢醫生進行進一步檢查和治療。"
        else:
             status = "餐後血糖可接受 (偏低)"
             advice = f"您的餐後血糖 ({val} mg/dL) 在可接受範圍但偏低，請注意觀察。"
    else:
        return "未知的血糖測量類型", "請指定 'fasting' (空腹) 或 'postprandial' (餐後)。", ""
    return status, advice, normal_range_info

def process_health_summary(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    # This function is not directly used in the current trend analysis flow,
    # but kept for completeness if other parts of the system use it.
    print("Warning: process_health_summary is likely not correctly aligned with current CSV column names (English).")
    return pd.DataFrame() # Return empty DataFrame as it's likely not working as expected with English CSVs.

def answer_care_question(user_question: str):
    if not user_question.strip():
        return "請輸入問題"
    if not api_key:
        return "錯誤：AI模型API金鑰未設定，無法回答問題。"
    try:
        model = genai.GenerativeModel(gemini_model)
        prompt = rag_prompt_template.format(question=user_question.strip())
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"抱歉，回答問題時發生錯誤: {e}"

def filter_data_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    # Ensure 'Date' column is used, not '日期'
    if 'Date' not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column.")
    
    df_filtered = df.copy()
    try:
        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['Date'])
    except Exception as e:
        raise ValueError(f"Error converting 'Date' column to datetime: {e}")

    if df_filtered.empty:
        return df_filtered

    df_filtered = df_filtered.sort_values(by="Date", ascending=True) # Sort ascending for trend plotting
    today = datetime.now().date()

    if period == 'today':
        return df_filtered[df_filtered['Date'].dt.date == today]
    elif period == '7days':
        seven_days_ago = today - timedelta(days=6)
        return df_filtered[(df_filtered['Date'].dt.date >= seven_days_ago) & (df_filtered['Date'].dt.date <= today)]
    elif period == '30days':
        thirty_days_ago = today - timedelta(days=29)
        return df_filtered[(df_filtered['Date'].dt.date >= thirty_days_ago) & (df_filtered['Date'].dt.date <= today)]
    elif period == 'all':
        return df_filtered
    else:
        print(f"Unrecognized time period '{period}', returning all data.")
        return df_filtered

def generate_trend_plot_base64(df: pd.DataFrame, columns_to_plot: list, ylabel: str, title: str):
    # This function generates a static PNG image (base64 encoded) for PDF embedding.
    if df.empty or not columns_to_plot:
        print(f"Debug: generate_trend_plot_base64 - DataFrame is empty or no columns to plot for '{title}'.")
        return None

    plot_df = df.copy()
    valid_columns_plotted = []

    for col in columns_to_plot:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
            if not plot_df[col].isnull().all():
                valid_columns_plotted.append(col)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame for plotting '{title}'.")

    if not valid_columns_plotted:
        print(f"Debug: generate_trend_plot_base64 - No valid numeric columns to plot for '{title}'.")
        return None

    if 'Date' not in plot_df.columns or plot_df['Date'].isnull().all():
        print(f"Debug: generate_trend_plot_base64 - 'Date' column is missing or all null for '{title}'.")
        return None
    
    plot_df = plot_df.dropna(subset=['Date'])
    if plot_df.empty:
        print(f"Debug: generate_trend_plot_base64 - DataFrame empty after dropping NaT Dates for '{title}'.")
        return None
        
    plot_df = plot_df.sort_values(by="Date")

    plt.figure(figsize=(12, 7))
    for col in valid_columns_plotted:
        if col in plot_df.columns and not plot_df[col].isnull().all():
            sns.lineplot(data=plot_df.dropna(subset=[col]), x="Date", y=col, label=col.replace('_', ' '), marker='o', linestyle='-')

    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("日期", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="指標", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    buffer = io.BytesIO()
    try:
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"Error saving trend plot for '{title}': {e}")
        return None
    finally:
        plt.close()

def generate_plotly_data(df: pd.DataFrame, columns_to_plot: list, ylabel: str, title: str):
    # This function generates Plotly-compatible JSON data for frontend rendering.
    if df.empty or not columns_to_plot:
        return {"data": [], "layout": {"title": "無數據可顯示"}}

    plot_df = df.copy()
    traces = []
    
    for col in columns_to_plot:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
            if not plot_df[col].isnull().all():
                trace = {
                    'x': plot_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                    'y': plot_df[col].tolist(),
                    'mode': 'lines+markers',
                    'name': col.replace('_', ' ')
                }
                traces.append(trace)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame for Plotly.")

    layout = {
        'title': title,
        'xaxis': {'title': '日期'},
        'yaxis': {'title': ylabel},
        'hovermode': 'x unified'
    }
    
    return {"data": traces, "layout": layout}


def generate_pca_plot(df: pd.DataFrame, numeric_cols: list, base_output_dir: str, title: str = "PCA 主成分分析"):
    # Adjusted path to use base_output_dir
    pca_output_folder = os.path.join(base_output_dir, "pca")
    os.makedirs(pca_output_folder, exist_ok=True)

    if df.empty or not numeric_cols or len(numeric_cols) < 2:
        print("Debug: generate_pca_plot - DataFrame empty or insufficient numeric columns for PCA.")
        return None
    
    data_for_pca = df[numeric_cols].copy()
    for col in numeric_cols:
        data_for_pca[col] = pd.to_numeric(data_for_pca[col], errors='coerce')
    
    data_for_pca = data_for_pca.dropna()

    if data_for_pca.shape[0] < 2 or data_for_pca.shape[1] < 2:
        print("Debug: generate_pca_plot - Insufficient data points or features after NaN removal for PCA.")
        return None
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_pca)
    
    n_components = min(2, data_for_pca.shape[0], data_for_pca.shape[1])
    if n_components < 1:
        print("Debug: generate_pca_plot - Cannot perform PCA with less than 1 component.")
        return None

    pca = SklearnPCA(n_components=n_components, random_state=42)
    principal_components = pca.fit_transform(scaled_data)
    
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=pc_cols)
    
    if 'Date' in df.columns and df.index.equals(data_for_pca.index):
        pca_df['Date'] = df.loc[data_for_pca.index, 'Date'].values
    else:
        pca_df['Hue'] = range(len(pca_df))

    plt.figure(figsize=(10, 7))
    hue_col = 'Date' if 'Date' in pca_df else 'Hue'
    
    if n_components == 1:
        sns.scatterplot(x=range(len(pca_df)), y='PC1', data=pca_df, hue=hue_col, palette='viridis', legend="auto", s=70)
        plt.xlabel('樣本索引', fontsize=12)
        plt.ylabel(f'主成分 1 (解釋變異: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    else:
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=hue_col, palette='viridis', legend="auto", s=70)
        plt.xlabel(f'主成分 1 (解釋變異: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        plt.ylabel(f'主成分 2 (解釋變異: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)

    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Use a unique filename for the PCA plot
    pca_plot_filename = f"pca_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    output_abs_path = os.path.join(pca_output_folder, pca_plot_filename)

    try:
        plt.savefig(output_abs_path)
        return output_abs_path
    except Exception as e:
        print(f"Error saving PCA plot: {e}")
        return None
    finally:
        plt.close()

def generate_trend_report_pdf(
    base_output_dir: str, # Changed from user_id
    request_timestamp_str: str,
    report_title: str, 
    time_period_label: str, 
    data_table_df: pd.DataFrame, 
    trend_plot_base64_data: str, # This is base64 string for embedding
    trend_analysis_text: str, 
    pca_plot_abs_path: str, # This is file path for wkhtmltopdf local access
    pca_interpretation_text: str,
    data_type_for_filename: str
) -> str:
    if not config:
        print("Error: wkhtmltopdf configuration not set. Cannot generate PDF report.")
        return None

    data_table_html = data_table_df.to_html(index=False, border=0, classes="dataframe") if not data_table_df.empty else "<p>此期間無數據可顯示。</p>"
    generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template = Template(PDF_REPORT_TEMPLATE)
    
    # Register markdown_to_html filter for this template instance
    template.environment.filters['markdown_to_html'] = markdown_to_html

    trend_plot_html_tag = f'<img src="data:image/png;base64,{trend_plot_base64_data}" alt="趨勢圖">' if trend_plot_base64_data else ''

    html_content = template.render(
        report_title=report_title,
        time_period_label=time_period_label,
        data_table_html=data_table_html,
        trend_plot_html_tag=trend_plot_html_tag, # Pass the img tag
        trend_analysis_text=trend_analysis_text,
        pca_plot_abs_path=pca_plot_abs_path, # Pass absolute path of PCA plot
        pca_interpretation_text=pca_interpretation_text,
        generation_timestamp=generation_timestamp
    )
    
    pdf_filename = f"{data_type_for_filename}_trend_report_{request_timestamp_str}.pdf"
    
    # Adjusted path to use base_output_dir
    pdf_output_folder = os.path.join(base_output_dir, "reports")
    os.makedirs(pdf_output_folder, exist_ok=True)
    abs_pdf_path = os.path.abspath(os.path.join(pdf_output_folder, pdf_filename))
    
    try:
        pdfkit.from_string(html_content, abs_pdf_path, configuration=config, options={'enable-local-file-access': ''})
        # Return relative path from static/ for Flask url_for
        return os.path.relpath(abs_pdf_path, os.path.abspath("static"))
    except Exception as e:
        print(f"Error generating PDF report '{pdf_filename}': {e}")
        # For debugging PDF generation issues, save the HTML content
        debug_html_path = os.path.join(os.path.dirname(abs_pdf_path), f"debug_trend_report_{request_timestamp_str}.html")
        with open(debug_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Debug HTML for PDF saved to: {debug_html_path}")
        raise # Re-raise the exception after saving debug info

def health_trend_analysis(csv_file_path: str, base_output_dir: str, analysis_timestamp_str: str, time_period_filter: str, data_type: str):
    try: # <-- This try block now correctly wraps the entire function logic
        # Check if file exists before reading
        if not os.path.exists(csv_file_path):
            return "錯誤：數據檔案不存在。", None, None, None, None

        # Use encoding='utf-8-sig' when reading the CSV
        df_original = pd.read_csv(csv_file_path, encoding='utf-8-sig')

        # Ensure 'Date' column is correctly identified and converted
        # Rename '日期' to 'Date' if present, otherwise assume 'Date' is already correct
        if '日期' in df_original.columns:
            df_original.rename(columns={'日期': 'Date'}, inplace=True)
        
        if 'Date' not in df_original.columns:
            raise ValueError("CSV 檔案中缺少 'Date' 欄位。")

        # Define numeric columns for each data type (these are CSV column names)
        numeric_cols_bp = ['Morning_Systolic', 'Morning_Diastolic', 'Morning_Pulse',
                           'Midday_Systolic', 'Midday_Diastolic', 'Midday_Pulse',
                           'Evening_Systolic', 'Evening_Diastolic', 'Evening_Pulse']
        numeric_cols_sugar = ['Morning_Fasting', 'Morning_Postprandial',
                              'Midday_Fasting', 'Midday_Postprandial', # Corrected from Noon_Fasting/Postprandial
                              'Evening_Fasting', 'Evening_Postprandial']

        cols_to_convert_numeric = []
        plot_columns_for_type = []
        if data_type == 'blood_pressure':
            cols_to_convert_numeric = numeric_cols_bp
            plot_columns_for_type = numeric_cols_bp
        elif data_type == 'blood_sugar':
            cols_to_convert_numeric = numeric_cols_sugar
            plot_columns_for_type = numeric_cols_sugar
        else:
            return "無效的數據類型。", None, None, None, None # Return None for plotly_data too

        df_analysis = df_original.copy()
        for col in cols_to_convert_numeric:
            if col in df_analysis.columns:
                df_analysis[col] = df_analysis[col].replace('無', np.nan)
                df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        
        # Filter data by period using the 'Date' column
        df_filtered = filter_data_by_period(df_analysis, time_period_filter)

        if df_filtered.empty:
            return "選定時間範圍內無數據可供分析。", None, None, None, None

        trend_plot_base64 = None # For PDF embedding
        plotly_data = None # For frontend Plotly chart
        pca_plot_file_path = None # For PDF embedding (file path)
        pca_interpretation = "PCA分析未執行或此報告不包含PCA。"

        time_period_labels = {
            'today': '當日', '7days': '最近7天', '30days': '最近30天', 'all': '所有歷史數據'
        }
        time_label = time_period_labels.get(time_period_filter, "指定期間")
        report_title_str = ""
        plot_ylabel = ""

        if data_type == 'blood_pressure':
            report_title_str = "血壓趨勢分析報告"
            plot_ylabel = "數值 (mmHg / 次/分)"
            pca_candidate_cols = [c for c in ['Morning_Systolic', 'Morning_Diastolic', 'Midday_Systolic', 'Midday_Diastolic', 'Evening_Systolic', 'Evening_Diastolic'] if c in df_filtered.columns and not df_filtered[c].isnull().all()]
            if len(pca_candidate_cols) >= 2:
                pca_plot_file_path = generate_pca_plot(df_filtered, pca_candidate_cols, base_output_dir, title="血壓主成分分析")
                if pca_plot_file_path:
                    pca_interpretation = (
                        "PCA (主成分分析) 圖表將多維健康數據（如不同時間點的血壓值）投影到二維平面上，"
                        "幫助視覺化數據點的分布和群集情況。圖中的每個點代表一次紀錄，相近的點表示健康狀況相似。"
                        "群集可能表示特定時期的健康狀況模式或變化趨勢。"
                    )
                else:
                    print(f"Warning: PCA plot generation failed for {data_type}.")
            else:
                print(f"Debug: Not enough valid numeric columns ({len(pca_candidate_cols)}) for PCA for {data_type}.")

        elif data_type == 'blood_sugar':
            report_title_str = "血糖趨勢分析報告"
            plot_ylabel = "血糖 (mg/dL)"
            pca_candidate_cols = [c for c in ['Morning_Fasting', 'Morning_Postprandial', 'Midday_Fasting', 'Midday_Postprandial', 'Evening_Fasting', 'Evening_Postprandial'] if c in df_filtered.columns and not df_filtered[c].isnull().all()]
            if len(pca_candidate_cols) >= 2:
                pca_plot_file_path = generate_pca_plot(df_filtered, pca_candidate_cols, base_output_dir, title="血糖主成分分析")
                if pca_plot_file_path:
                    pca_interpretation = (
                        "PCA (主成分分析) 圖表將多維健康數據（如不同時間點的血糖值）投影到二維平面上，"
                        "幫助視覺化數據點的分布和群集情況。圖中的每個點代表一次紀錄，相近的點表示健康狀況相似。"
                        "群集可能表示特定時期的健康狀況模式或變化趨勢。"
                    )
                else:
                    print(f"Warning: PCA plot generation failed for {data_type}.")
            else:
                print(f"Debug: Not enough valid numeric columns ({len(pca_candidate_cols)}) for PCA for {data_type}.")
        
        actual_cols_to_plot = [col for col in plot_columns_for_type if col in df_filtered.columns and not df_filtered[col].isnull().all()]

        if actual_cols_to_plot:
            # Generate base64 image for PDF
            trend_plot_base64 = generate_trend_plot_base64(
                df_filtered,
                columns_to_plot=actual_cols_to_plot,
                ylabel=plot_ylabel,
                title=f"{report_title_str} ({time_label})"
            )
            # Generate Plotly data for frontend
            plotly_data = generate_plotly_data(
                df_filtered,
                columns_to_plot=actual_cols_to_plot,
                ylabel=plot_ylabel,
                title=f"{report_title_str} ({time_label})"
            )
        else:
            print(f"Warning: No actual columns to plot for {data_type}.")

        trend_analysis_output_text = "AI趨勢分析未能生成。"
        if api_key:
            try:
                prompt_df_display = df_filtered[['Date'] + actual_cols_to_plot].copy()
                prompt_df_display['Date'] = prompt_df_display['Date'].dt.strftime('%Y-%m-%d')
                for col in actual_cols_to_plot:
                    prompt_df_display[col] = prompt_df_display[col].apply(lambda x: '無' if pd.isna(x) else x)

                data_for_prompt = prompt_df_display.to_string(index=False, na_rep='無')
                
                model = genai.GenerativeModel(gemini_model)
                full_trend_prompt = f"{trend_prompt}\n以下是分析數據 ({time_label}):\n{data_for_prompt}"
                response = model.generate_content(full_trend_prompt)
                trend_analysis_output_text = response.text.strip()
            except Exception as gemini_err:
                print(f"Error calling Gemini for trend analysis: {gemini_err}")
                trend_analysis_output_text = f"AI趨勢分析時發生錯誤: {gemini_err}"
        else:
            trend_analysis_output_text = "AI模型API金鑰未設定，無法執行AI趨勢分析。"

        pdf_table_df_display = df_filtered[['Date'] + actual_cols_to_plot].copy()
        pdf_table_df_display['Date'] = pdf_table_df_display['Date'].dt.strftime('%Y-%m-%d')
        for col in actual_cols_to_plot:
            pdf_table_df_display[col] = pdf_table_df_display[col].apply(lambda x: '無' if pd.isna(x) else f"{x:.0f}" if isinstance(x, float) and x.is_integer() else x)

        pdf_report_rel_static_path = None
        if config and trend_plot_base64: # Only generate PDF if wkhtmltopdf is configured AND base64 image was generated
            pdf_report_rel_static_path = generate_trend_report_pdf(
                base_output_dir=base_output_dir, # Pass base_output_dir
                request_timestamp_str=analysis_timestamp_str,
                report_title=report_title_str,
                time_period_label=time_label,
                data_table_df=pdf_table_df_display.drop(columns=[col for col in pdf_table_df_display.columns if 'Unnamed' in str(col)], errors='ignore'),
                trend_plot_base64_data=trend_plot_base64, # Pass base64 string
                trend_analysis_text=trend_analysis_output_text,
                pca_plot_abs_path=pca_plot_file_path, # Pass absolute path of PCA plot
                pca_interpretation_text=pca_interpretation,
                data_type_for_filename=data_type 
            )
        else:
            print("Skipping PDF generation as wkhtmltopdf is not configured or trend plot failed.")
            trend_analysis_output_text += "\n(PDF報告生成已跳過，因系統未配置PDF引擎或趨勢圖生成失敗)"
        
        # Save trend image to a file for direct download if base64 was generated
        trend_image_rel_static_path = None
        if trend_plot_base64:
            # Adjusted path to use base_output_dir
            img_folder_abs = os.path.join(base_output_dir, "trend")
            os.makedirs(img_folder_abs, exist_ok=True)
            
            img_filename = f"{data_type}_trend_{analysis_timestamp_str}.png"
            img_abs_path = os.path.join(img_folder_abs, img_filename)
            try:
                with open(img_abs_path, "wb") as fh:
                    fh.write(base64.b64decode(trend_plot_base64))
                # Return relative path from static/ for Flask url_for
                trend_image_rel_static_path = os.path.relpath(img_abs_path, os.path.abspath("static"))
            except Exception as img_save_err:
                print(f"Error saving trend image {img_filename}: {img_save_err}")

        # Return all necessary data
        return trend_analysis_output_text, trend_image_rel_static_path, pdf_report_rel_static_path, plotly_data

    except Exception as e: # <-- This except block now correctly wraps the entire function logic
        import traceback
        traceback.print_exc()
        # Ensure consistent return signature even on error
        return f"趨勢分析主流程發生錯誤: {str(e)}", None, None, None, None

# The __main__ block is for testing health_analysis.py directly, not part of the Flask app.
# It needs to be updated to reflect the new return values of health_trend_analysis.
if __name__ == '__main__':
    # This is a local helper for the __main__ block only, not the global one in app.py
    def _local_clear_user_data_folder(user_id: str, subfolder: str):
        folder_path = os.path.join("static", "user_data", user_id, subfolder)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Cleared folder: {folder_path}")
            except OSError as e:
                print(f"Error clearing folder {folder_path}: {e}")
        os.makedirs(folder_path, exist_ok=True) # Recreate the folder

    user_test_id = "testuser123" # This ID should match a user in your auth.py or a test user
    # Define base_output_dir for testing within health_analysis.py
    base_output_dir_for_test = os.path.join("static", "user_data", user_test_id)
    os.makedirs(os.path.join(base_output_dir_for_test, "data"), exist_ok=True) # Ensure data folder exists
    
    # Clear and recreate specific user data folders for testing
    _local_clear_user_data_folder(user_test_id, "trend")
    _local_clear_user_data_folder(user_test_id, "pca")
    _local_clear_user_data_folder(user_test_id, "reports")
    _local_clear_user_data_folder(user_test_id, "summary")

    dummy_data_path = os.path.join(base_output_dir_for_test, "data", "dummy_health_data.csv")

    # Dummy BP data (using English column names as expected by validate_bp_csv)
    # Ensure enough data points for PCA (at least 2 rows after filtering and dropping NaNs)
    bp_data = {
        'Date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)], # Reduced to 10 days for simpler test
        'Morning_Systolic': [120+i%10 - 5 for i in range(10)],
        'Morning_Diastolic': [80+i%5 - 2 for i in range(10)],
        'Morning_Pulse': [70 + i % 5 for i in range(10)],
        'Midday_Systolic': [125+i%10 - 5 for i in range(10)],
        'Midday_Diastolic': [78+i%5 - 2 for i in range(10)],
        'Midday_Pulse': [72 + i % 5 for i in range(10)],
        'Evening_Systolic': [130+i%12 - 6 for i in range(10)],
        'Evening_Diastolic': [82+i%6 - 3 for i in range(10)],
        'Evening_Pulse': [68 + i % 5 for i in range(10)]
    }
    dummy_bp_df = pd.DataFrame(bp_data)
    dummy_bp_df.to_csv(os.path.join(base_output_dir_for_test, "data", "dummy_health_data.csv"), index=False, encoding='utf-8-sig')

    print(f"\n--- Testing Blood Pressure Trend Analysis (7 days) ---")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") # Changed to _ for consistency
    # Corrected call: removed 'health_analysis.' prefix
    bp_analysis_text, bp_image_url, bp_pdf_url, bp_plotly_data = health_trend_analysis(os.path.join(base_output_dir_for_test, "data", "dummy_health_data.csv"), base_output_dir_for_test, ts, "7days", "blood_pressure")
    if bp_analysis_text and "錯誤" not in bp_analysis_text: # Check for error message in text
        print(f"BP Analysis Text:\n{bp_analysis_text[:200]}...")
        print(f"BP Image URL (relative to static): {bp_image_url}")
        print(f"BP PDF Report URL (relative to static): {bp_pdf_url}")
        print(f"BP Plotly Data (first trace): {bp_plotly_data['data'][0] if bp_plotly_data and bp_plotly_data['data'] else 'N/A'}")
    else:
        print(f"Error in BP analysis: {bp_analysis_text}")

    # Dummy Sugar data (using English column names)
    sugar_data = {
        'Date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)], # Reduced to 10 days for simpler test
        'Morning_Fasting': [90+i%8 - 4 for i in range(10)],
        'Morning_Postprandial': [130+i%20 - 10 for i in range(10)],
        'Midday_Fasting': [95+i%8 - 4 for i in range(10)],
        'Midday_Postprandial': [135+i%20 - 10 for i in range(10)],
        'Evening_Fasting': [92+i%8 - 4 for i in range(10)],
        'Evening_Postprandial': [140+i%20 - 10 for i in range(10)]
    }
    dummy_sugar_df = pd.DataFrame(sugar_data)
    dummy_sugar_df.to_csv(os.path.join(base_output_dir_for_test, "data", "dummy_health_data.csv"), index=False, encoding='utf-8-sig')

    print(f"\n--- Testing Blood Sugar Trend Analysis (30 days) ---")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") # Changed to _ for consistency
    # Corrected call: removed 'health_analysis.' prefix
    sugar_analysis_text, sugar_image_url, sugar_pdf_url, sugar_plotly_data = health_trend_analysis(os.path.join(base_output_dir_for_test, "data", "dummy_health_data.csv"), base_output_dir_for_test, ts, "30days", "blood_sugar")
    if sugar_analysis_text and "錯誤" not in sugar_analysis_text:
        print(f"Sugar Analysis Text:\n{sugar_analysis_text[:200]}...")
        print(f"Sugar Image URL (relative to static): {sugar_image_url}")
        print(f"Sugar PDF Report URL (relative to static): {sugar_pdf_url}")
        print(f"Sugar Plotly Data (first trace): {sugar_plotly_data['data'][0] if sugar_plotly_data and sugar_plotly_data['data'] else 'N/A'}")
    else:
        print(f"Error in Sugar analysis: {sugar_analysis_text}")

    print(f"\nNote: Dummy CSV '{os.path.join(base_output_dir_for_test, 'data', 'dummy_health_data.csv')}' may need to be manually removed.")