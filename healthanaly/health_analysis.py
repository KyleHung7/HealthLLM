import os
import pandas as pd
import google.generativeai as genai
import pdfkit
from jinja2 import Template
from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from lib import mdToHtml, clear_user_data_folder  # Assuming these are correctly defined elsewhere

# For PCA and date handling
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA
import numpy as np
from datetime import datetime, timedelta, timezone

# Configure matplotlib
matplotlib.use('Agg')
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Warning: Could not set preferred CJK fonts: {e}")

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-1.5-flash")
genai.configure(api_key=api_key)

# Configure wkhtmltopdf
WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH")
if not WKHTMLTOPDF_PATH or not os.path.exists(WKHTMLTOPDF_PATH):
    raise EnvironmentError("WKHTMLTOPDF_PATH is not set or the wkhtmltopdf executable does not exist.")
config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

# Blood Pressure and Blood Sugar Standards
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

# Prompts (simplified since we'll use analyze functions instead of LLM for standards)
blood_pressure_prompt = """
你是一位長照輔助分析專家，請根據以下長者每日的血壓紀錄，提供簡潔的健康摘要與建議。
請輸出下列表格格式：
| 日期 | 早上收縮壓 (mmHg) | 早上舒張壓 (mmHg) | 早上脈搏 (次/分鐘) | 中午收縮壓 (mmHg) | 中午舒張壓 (mmHg) | 中午脈搏 (次/分鐘) | 晚上收縮壓 (mmHg) | 晚上舒張壓 (mmHg) | 晚上脈搏 (次/分鐘) | 達標狀況 | 養護建議 |
|------|-------------------|-------------------|---------------------|-------------------|-------------------|---------------------|-------------------|-------------------|---------------------|-----------|----------|
"""

blood_sugar_prompt = """
你是一位長照輔助分析專家，請根據以下長者每日的血糖紀錄，提供簡潔的健康摘要與建議。
請輸出下列表格格式：
| 日期 | 早餐前血糖 (mg/dL) | 早餐後血糖 (mg/dL) | 午餐前血糖 (mg/dL) | 午餐後血糖 (mg/dL) | 晚餐前血糖 (mg/dL) | 晚餐後血糖 (mg/dL) | 達標狀況 | 養護建議 |
|------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-----------|----------|
"""

trend_prompt = """
你是一位健康數據分析師，請根據以下血壓或血糖紀錄（已根據用戶選擇的時間區間篩選），分析是否出現異常趨勢（如連續升高、波動劇烈等），並提供簡短建議。
請輸出格式如下：
- 🟡 指標變化：...
- 🔴 建議：...
"""

rag_prompt_template = """
你是長照照護助手，請根據你掌握的知識，針對照顧者的問題提供具體建議。
問題：「{question}」
"""

# HTML Template for PDF Report (unchanged)
PDF_REPORT_TEMPLATE = """
<html>
<head>
    <meta charset="utf-8">
    <title>{{ report_title }}</title>
    <style>
        body { font-family: 'Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', sans-serif; margin: 20px; color: #333; }
        h1, h2, h3 { color: #0056b3; }
        h1 { text-align: center; margin-bottom: 30px; }
        h2 { border-bottom: 2px solid #0056b3; padding-bottom: 5px; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        img { max-width: 90%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; padding: 5px; }
        .analysis-section { margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }
        .analysis-section p { white-space: pre-wrap; line-height: 1.6; }
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

    {% if trend_plot_abs_path %}
    <h2>趨勢圖</h2>
    <img src="file:///{{ trend_plot_abs_path }}" alt="趨勢圖">
    {% endif %}

    {% if trend_analysis_text %}
    <h2>趨勢分析與建議 (AI生成)</h2>
    <div class="analysis-section">
        <p>{{ trend_analysis_text }}</p>
    </div>
    {% endif %}

    {% if pca_plot_abs_path %}
    <h2>PCA 主成分分析圖</h2>
    <img src="file:///{{ pca_plot_abs_path }}" alt="PCA 分析圖">
    {% endif %}
    
    {% if pca_interpretation_text %}
    <h2>PCA 分析說明</h2>
    <div class="analysis-section">
        <p>{{ pca_interpretation_text }}</p>
    </div>
    {% endif %}

    <div class="footer">
        <p>HealthLLM 健康報告 - 生成時間: {{ generation_timestamp }}</p>
    </div>
</body>
</html>
"""

# Helper Functions (mostly unchanged, except for validation and summary processing)
def parse_markdown_table(markdown_text: str) -> pd.DataFrame:
    lines = [line.strip() for line in markdown_text.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if not table_lines or len(table_lines) < 3:
        print("Debug: parse_markdown_table - Not enough lines for a table.")
        return None 
    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    if not all(set(s.strip()) <= set('-| ') for s in table_lines[1].strip("|").split("|")):
        print("Debug: parse_markdown_table - Separator line is not valid.")
        return None
    data_rows = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == len(headers):
            data_rows.append(cells)
        else:
            print(f"Debug: parse_markdown_table - Mismatch in cell count for row: {line}.")
            continue 
    if not data_rows:
        print("Debug: parse_markdown_table - No valid data rows found.")
        return None
    return pd.DataFrame(data_rows, columns=headers)

def generate_html_for_summary(df: pd.DataFrame, title="健康紀錄分析") -> str:
    original_html_template = """
    <html><head><meta charset="utf-8"><style>body { font-family: Arial, sans-serif; } table { border-collapse: collapse; width: 100%; } th, td { border: 1px solid black; padding: 8px; text-align: left; } th { background-color: #f2f2f2; } img { max-width: 100%; margin-top: 20px; }</style></head>
    <body><h2>{{ title }}</h2><table><thead><tr>{% for col in table.columns %}<th>{{ col }}</th>{% endfor %}</tr></thead><tbody>{% for row in table.values %}<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}</tbody></table></body></html>
    """
    template = Template(original_html_template)
    return template.render(table=df, title=title)

def generate_pdf_from_html_summary(html_content: str, user_id: str, pdf_filename: str) -> str:
    pdf_path = f"static/{user_id}/summary/{pdf_filename}"
    abs_pdf_path = os.path.abspath(pdf_path)
    os.makedirs(os.path.dirname(abs_pdf_path), exist_ok=True)
    pdfkit.from_string(html_content, abs_pdf_path, configuration=config, options={'enable-local-file-access': ''})
    return pdf_path

def validate_bp_csv(df):
    required_columns = [
        '日期', '早上收縮壓 (mmHg)', '早上舒張壓 (mmHg)', '早上脈搏 (次/分鐘)',
        '中午收縮壓 (mmHg)', '中午舒張壓 (mmHg)', '中午脈搏 (次/分鐘)',
        '晚上收縮壓 (mmHg)', '晚上舒張壓 (mmHg)', '晚上脈搏 (次/分鐘)'
    ]
    df_columns_normalized = [str(col).strip().lower() for col in df.columns]
    required_columns_normalized = [col.strip().lower() for col in required_columns]
    present = all(col_req in df_columns_normalized for col_req in required_columns_normalized)
    if not present:
        missing = [col for col in required_columns_normalized if col not in df_columns_normalized]
        print(f"Debug: validate_bp_csv - Missing BP columns: {missing}")
    return present

def validate_sugar_csv(df):
    required_columns = [
        '日期', '早上空腹血糖 (mg/dL)', '早上餐後血糖 (mg/dL)',
        '中午空腹血糖 (mg/dL)', '中午餐後血糖 (mg/dL)',
        '晚餐前血糖 (mg/dL)', '晚餐後血糖 (mg/dL)'
    ]
    df_columns_normalized = [str(col).strip().lower() for col in df.columns]
    required_columns_normalized = [col.strip().lower() for col in required_columns]
    present = all(col_req in df_columns_normalized for col_req in required_columns_normalized)
    if not present:
        missing = [col for col in required_columns_normalized if col not in df_columns_normalized]
        print(f"Debug: validate_sugar_csv - Missing Sugar columns: {missing}")
    return present

def analyze_blood_pressure(systolic, diastolic, pulse=None):
    if not (isinstance(systolic, (int, float)) and isinstance(diastolic, (int, float))):
        return "血壓輸入無效", "請輸入有效的數字作為血壓值。", ""
    if pulse is not None and not isinstance(pulse, (int, float)):
        return "脈搏輸入無效", "請輸入有效的數字作為脈搏值，或留空。", ""
    s = int(systolic)
    d = int(diastolic)
    status = "未知血壓狀態"
    advice = "請諮詢醫生以獲得專業評估。"
    normal_range_info = f"理想血壓: 收縮壓 < {BP_NORMAL_SYSTOLIC_MAX + 1} mmHg 且 舒張壓 < {BP_NORMAL_DIASTOLIC_MAX + 1} mmHg。"
    if s > BP_CRISIS_SYSTOLIC_MIN or d > BP_CRISIS_DIASTOLIC_MIN:
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
            advice = "您的血壓可能偏低，如果伴有頭暈、乏力等症狀，請諮詢醫生。"
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
            status = "血壓數據異常"
            advice = "您的血壓數據組合較為特殊或不完整，建議諮詢醫生。"
    if pulse is not None:
        p = int(pulse)
        pulse_info = f"脈搏: {p} 次/分. "
        if not (60 <= p <= 100):
            pulse_info += "脈搏速率不在常規靜息範圍 (60-100 次/分)，建議注意。"
        advice = pulse_info + advice
    return status, advice, normal_range_info

def analyze_blood_sugar(value, measurement_type="fasting"):
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
        elif BS_NORMAL_FASTING_MIN <= val <= BS_NORMAL_FASTING_MAX:
            status = "正常空腹血糖"
            advice = f"您的空腹血糖 ({val} mg/dL) 在理想範圍。"
        elif BS_PREDIABETES_FASTING_MIN <= val <= BS_PREDIABETES_FASTING_MAX:
            status = "糖尿病前期 (空腹)"
            advice = f"您的空腹血糖 ({val} mg/dL) 偏高，屬於糖尿病前期。建議改善飲食、增加運動，並定期追蹤血糖。"
        elif val >= BS_DIABETES_FASTING_MIN:
            status = "糖尿病 (、空腹)"
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
            status = "餐後血糖可接受"
            advice = f"您的餐後血糖 ({val} mg/dL) 在可接受範圍，但請持續監測。"
    else:
        return "未知的血糖測量類型", "請指定 'fasting' (空腹) 或 'postprandial' (餐後)。", ""
    return status, advice, normal_range_info

def process_health_summary(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    df_copy = df.copy()
    if '日期' in df_copy.columns:
        df_copy['日期'] = pd.to_datetime(df_copy['日期']).dt.strftime('%Y-%m-%d')
    df_copy = df_copy.fillna("無")
    result_rows = []

    if data_type == 'blood_pressure':
        time_slots = ['早上', '中午', '晚上']
        for _, row in df_copy.iterrows():
            for time in time_slots:
                sys_col = f'{time}收縮壓 (mmHg)'
                dia_col = f'{time}舒張壓 (mmHg)'
                pulse_col = f'{time}脈搏 (次/分鐘)'
                if sys_col in row and dia_col in row and row[sys_col] != '無' and row[dia_col] != '無':
                    try:
                        systolic = float(row[sys_col])
                        diastolic = float(row[dia_col])
                        pulse = float(row[pulse_col]) if pulse_col in row and row[pulse_col] != '無' else None
                        status, advice, _ = analyze_blood_pressure(systolic, diastolic, pulse)
                        result_rows.append({
                            '日期': row['日期'],
                            f'{time}收縮壓 (mmHg)': row[sys_col],
                            f'{time}舒張壓 (mmHg)': row[dia_col],
                            f'{time}脈搏 (次/分鐘)': row[pulse_col] if pulse_col in row else '無',
                            '達標狀況': status,
                            '養護建議': advice
                        })
                    except (ValueError, TypeError):
                        continue
    else:  # blood_sugar
        time_slots = [('早餐前', '早上空腹', 'fasting'), ('早餐後', '早上餐後', 'postprandial'),
                      ('午餐前', '中午空腹', 'fasting'), ('午餐後', '中午餐後', 'postprandial'),
                      ('晚餐前', '晚餐前', 'fasting'), ('晚餐後', '晚餐後', 'postprandial')]
        for _, row in df_copy.iterrows():
            for display, col_prefix, measure_type in time_slots:
                col = f'{col_prefix}血糖 (mg/dL)'
                if col in row and row[col] != '無':
                    try:
                        value = float(row[col])
                        status, advice, _ = analyze_blood_sugar(value, measure_type)
                        result_rows.append({
                            '日期': row['日期'],
                            f'{display}血糖 (mg/dL)': row[col],
                            '達標狀況': status,
                            '養護建議': advice
                        })
                    except (ValueError, TypeError):
                        continue

    if not result_rows:
        print(f"Debug: process_health_summary - No valid data processed for {data_type}.")
        return pd.DataFrame()
    return pd.DataFrame(result_rows)

def answer_care_question(user_question):
    if not user_question.strip():
        return "請輸入問題"
    model = genai.GenerativeModel(gemini_model)
    prompt = rag_prompt_template.format(question=user_question.strip())
    response = model.generate_content(prompt)
    return response.text.strip()

def filter_data_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if '日期' not in df.columns:
        raise ValueError("DataFrame must contain a '日期' column.")
    try:
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.dropna(subset=['日期'])
    except Exception as e:
        print(f"Error converting '日期' column to datetime: {e}")
        return pd.DataFrame(columns=df.columns)
    df = df.sort_values(by="日期", ascending=False)
    today = datetime.now().date()
    if period == 'today':
        return df[df['日期'].dt.date == today]
    elif period == '7days':
        seven_days_ago = today - timedelta(days=6)
        return df[(df['日期'].dt.date >= seven_days_ago) & (df['日期'].dt.date <= today)]
    elif period == '30days':
        thirty_days_ago = today - timedelta(days=29)
        return df[(df['日期'].dt.date >= thirty_days_ago) & (df['日期'].dt.date <= today)]
    else:
        return df

def generate_trend_plot_for_pdf(df: pd.DataFrame, output_abs_path: str, columns_to_plot: list, ylabel: str, title: str):
    if df.empty or not columns_to_plot:
        print(f"Debug: generate_trend_plot_for_pdf - DataFrame is empty or no columns to plot for '{title}'.")
        return None
    os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)
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
        print(f"Debug: generate_trend_plot_for_pdf - No valid numeric columns to plot for '{title}'.")
        return None
    if '日期' not in plot_df.columns or plot_df['日期'].isnull().all():
        print(f"Debug: generate_trend_plot_for_pdf - '日期' column is missing or all null for '{title}'.")
        return None
    plot_df = plot_df.dropna(subset=['日期'])
    plot_df = plot_df.sort_values(by="日期")
    plt.figure(figsize=(12, 6))
    for col in valid_columns_plotted:
        if col in plot_df.columns and not plot_df[col].isnull().all():
            sns.lineplot(data=plot_df.dropna(subset=[col]), x="日期", y=col, label=col, marker='o', linestyle='-')
    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("日期", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="指標")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(output_abs_path)
        plt.close()
        return output_abs_path
    except Exception as e:
        print(f"Error saving trend plot for '{title}' at {output_abs_path}: {e}")
        plt.close()
        return None

def generate_pca_plot(df: pd.DataFrame, numeric_cols: list, output_abs_path: str, title: str = "PCA 主成分分析"):
    if df.empty or not numeric_cols or len(numeric_cols) < 2:
        print("Debug: generate_pca_plot - DataFrame empty or insufficient numeric columns for PCA.")
        return None
    os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)
    data_for_pca = df[numeric_cols].copy()
    for col in numeric_cols:
        data_for_pca[col] = pd.to_numeric(data_for_pca[col], errors='coerce')
    data_for_pca = data_for_pca.dropna()
    if data_for_pca.shape[0] < 2 or data_for_pca.shape[1] < 2:
        print("Debug: generate_pca_plot - Insufficient data points or features after NaN removal for PCA.")
        return None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_pca)
    pca = SklearnPCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    if '日期' in df.columns and df.index.equals(data_for_pca.index):
        pca_df['日期'] = df.loc[data_for_pca.index, '日期']
    else:
        pca_df['日期'] = range(len(pca_df))
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue='日期', palette='viridis', legend=None, s=70)
    plt.title(title, fontsize=16)
    plt.xlabel(f'主成分 1 (解釋變異: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'主成分 2 (解釋變異: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(output_abs_path)
        plt.close()
        return output_abs_path
    except Exception as e:
        print(f"Error saving PCA plot: {e}")
        plt.close()
        return None

def generate_trend_report_pdf(
    user_id: str, 
    request_timestamp_str: str,
    report_title: str, 
    time_period_label: str, 
    data_table_df: pd.DataFrame, 
    trend_plot_abs_path: str, 
    trend_analysis_text: str, 
    pca_plot_abs_path: str, 
    pca_interpretation_text: str
) -> str:
    data_table_html = data_table_df.to_html(index=False, border=0, classes="dataframe") if not data_table_df.empty else None
    generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template = Template(PDF_REPORT_TEMPLATE)
    html_content = template.render(
        report_title=report_title,
        time_period_label=time_period_label,
        data_table_html=data_table_html,
        trend_plot_abs_path=trend_plot_abs_path,
        trend_analysis_text=trend_analysis_text,
        pca_plot_abs_path=pca_plot_abs_path,
        pca_interpretation_text=pca_interpretation_text,
        generation_timestamp=generation_timestamp
    )
    pdf_filename = f"trend_report_{request_timestamp_str}.pdf"
    relative_pdf_path = os.path.join(user_id, "reports", pdf_filename)
    static_pdf_path = os.path.join("static", relative_pdf_path)
    abs_pdf_path = os.path.abspath(static_pdf_path)
    os.makedirs(os.path.dirname(abs_pdf_path), exist_ok=True)
    try:
        pdfkit.from_string(html_content, abs_pdf_path, configuration=config, options={'enable-local-file-access': ''})
        return relative_pdf_path
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        debug_html_path = os.path.join(os.path.dirname(abs_pdf_path), f"debug_trend_report_{request_timestamp_str}.html")
        with open(debug_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Debug HTML for PDF saved to: {debug_html_path}")
        raise

def health_trend_analysis(file_path: str, user_id: str, request_timestamp_str: str, time_period: str):
    if not os.path.exists(file_path):
        return {"error": "請先上傳或確認 CSV 檔案路徑正確。"}
    try:
        df_original = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"讀取 CSV 檔案失敗: {e}"}
    if '日期' not in df_original.columns:
        return {"error": "CSV 檔案中缺少 '日期' 欄位。"}
    df_original.rename(columns={col: '日期' for col in df_original.columns if 'date' in col.lower() or '日期' in col}, inplace=True)
    try:
        df_filtered = filter_data_by_period(df_original.copy(), time_period)
    except ValueError as e:
        return {"error": str(e)}
    if df_filtered.empty:
        return {"error": f"在選定的 '{time_period}' 時間區間內沒有找到數據。"}
    data_type = None
    value_columns = []
    numeric_cols_for_pca = []
    plot_ylabel = ""
    report_main_title = ""
    df_filtered.columns = [str(col).strip() for col in df_filtered.columns]
    if validate_bp_csv(df_filtered):
        data_type = 'blood_pressure'
        report_main_title = "血壓趨勢分析報告"
        value_columns = [
            '早上收縮壓 (mmHg)', '早上舒張壓 (mmHg)', '早上脈搏 (次/分鐘)',
            '中午收縮壓 (mmHg)', '中午舒張壓 (mmHg)', '中午脈搏 (次/分鐘)',
            '晚上收縮壓 (mmHg)', '晚上舒張壓 (mmHg)', '晚上脈搏 (次/分鐘)'
        ]
        numeric_cols_for_pca = [col for col in value_columns if col in df_filtered.columns]
        plot_ylabel = "血壓 (mmHg) / 脈搏 (次/分鐘)"
    elif validate_sugar_csv(df_filtered):
        data_type = 'blood_sugar'
        report_main_title = "血糖趨勢分析報告"
        value_columns = [
            '早上空腹血糖 (mg/dL)', '早上餐後血糖 (mg/dL)',
            '中午空腹血糖 (mg/dL)', '中午餐後血糖 (mg/dL)',
            '晚餐前血糖 (mg/dL)', '晚餐後血糖 (mg/dL)'
        ]
        numeric_cols_for_pca = [col for col in value_columns if col in df_filtered.columns]
        plot_ylabel = "血糖 (mg/dL)"
    else:
        return {"error": "CSV 檔案格式不符合血壓或血糖分析要求，或篩選後數據欄位不足。"}
    trend_plot_filename = f"{data_type}_trend_{request_timestamp_str}.png"
    trend_plot_rel_path_for_url = os.path.join(user_id, "trend_plots", trend_plot_filename)
    trend_plot_static_path = os.path.join("static", trend_plot_rel_path_for_url)
    trend_plot_abs_path = os.path.abspath(trend_plot_static_path)
    actual_cols_to_plot = [col for col in value_columns if col in df_filtered.columns]
    trend_plot_final_path = generate_trend_plot_for_pdf(
        df_filtered.copy(), 
        trend_plot_abs_path, 
        actual_cols_to_plot, 
        plot_ylabel, 
        f"{report_main_title} - 趨勢圖"
    )
    if not trend_plot_final_path:
        print(f"Warning: Trend plot generation failed for {data_type}.")
        trend_plot_abs_path = None
    model = genai.GenerativeModel(gemini_model)
    llm_df = df_filtered.copy()
    if '日期' in llm_df.columns:
        llm_df['日期'] = pd.to_datetime(llm_df['日期']).dt.strftime('%Y-%m-%d')
    for col in numeric_cols_for_pca:
        if col in llm_df.columns:
            llm_df[col] = pd.to_numeric(llm_df[col], errors='coerce').fillna('無')
            llm_df[col] = llm_df[col].astype(str)
    llm_content = llm_df[['日期'] + [col for col in actual_cols_to_plot if col in llm_df.columns]].to_csv(index=False)
    trend_analysis_text = "AI 趨勢分析無法生成。"
    try:
        response = model.generate_content(f"{trend_prompt}\n\n{llm_content}")
        trend_analysis_text = response.text.strip()
    except Exception as e:
        print(f"Error getting LLM trend analysis: {e}")
        trend_analysis_text = f"AI 趨勢分析無法生成：{e}"
    pca_plot_abs_path = None
    pca_interpretation_text = "此期間數據不足或格式不符，無法進行 PCA 分析。"
    pca_input_df = df_filtered.copy()
    valid_numeric_cols_for_pca = []
    for col in numeric_cols_for_pca:
        if col in pca_input_df.columns:
            pca_input_df[col] = pd.to_numeric(pca_input_df[col], errors='coerce')
            if not pca_input_df[col].isnull().all():
                valid_numeric_cols_for_pca.append(col)
    if valid_numeric_cols_for_pca and len(valid_numeric_cols_for_pca) >= 2:
        pca_input_df_cleaned = pca_input_df[['日期'] + valid_numeric_cols_for_pca].dropna(subset=valid_numeric_cols_for_pca)
        if pca_input_df_cleaned.shape[0] >= 2:
            pca_plot_filename = f"pca_plot_{request_timestamp_str}.png"
            pca_plot_rel_path_for_url = os.path.join(user_id, "pca_plots", pca_plot_filename)
            pca_plot_static_path = os.path.join("static", pca_plot_rel_path_for_url)
            pca_plot_abs_path_candidate = os.path.abspath(pca_plot_static_path)
            pca_plot_final_path = generate_pca_plot(
                pca_input_df_cleaned,
                valid_numeric_cols_for_pca, 
                pca_plot_abs_path_candidate,
                f"{report_main_title} - PCA分群圖"
            )
            if pca_plot_final_path:
                pca_plot_abs_path = pca_plot_final_path
                pca_interpretation_text = (
                    "PCA (主成分分析) 圖表將多維健康數據（如不同時間點的血壓/血糖值）投影到二維平面上，"
                    "幫助視覺化數據點的分布和群集情況。圖中的每個點代表一次紀錄，相近的點表示健康狀況相似。"
                    "群集可能表示特定時期的健康狀況模式或變化趨勢。"
                )
            else:
                print(f"Warning: PCA plot generation failed for {data_type}.")
        else:
            print(f"Debug: Not enough valid data rows ({pca_input_df_cleaned.shape[0]}) for PCA after cleaning.")
    else:
        print(f"Debug: Not enough valid numeric columns ({len(valid_numeric_cols_for_pca)}) for PCA.")
    time_period_labels = {
        "today": "當日", "7days": "最近 7 天", "30days": "最近 30 天", "all": "所有歷史數據"
    }
    time_label = time_period_labels.get(time_period, time_period)
    try:
        pdf_report_relative_path = generate_trend_report_pdf(
            user_id=user_id,
            request_timestamp_str=request_timestamp_str,
            report_title=report_main_title,
            time_period_label=time_label,
            data_table_df=df_filtered.drop(columns=[col for col in df_filtered.columns if 'Unnamed' in str(col)], errors='ignore'),
            trend_plot_abs_path=trend_plot_abs_path,
            trend_analysis_text=trend_analysis_text,
            pca_plot_abs_path=pca_plot_abs_path,
            pca_interpretation_text=pca_interpretation_text
        )
    except Exception as e:
        return {"error": f"產生 PDF 報告失敗: {e}"}
    return {
        "preview_plot_url": trend_plot_rel_path_for_url if trend_plot_abs_path else None,
        "trend_analysis_html": mdToHtml(trend_analysis_text),
        "pdf_report_url": pdf_report_relative_path,
        "error": None
    }

if __name__ == '__main__':
    user_test_id = "testuser123"
    dummy_data_path = "dummy_health_data.csv"
    os.makedirs(f"static/{user_test_id}/trend_plots", exist_ok=True)
    os.makedirs(f"static/{user_test_id}/pca_plots", exist_ok=True)
    os.makedirs(f"static/{user_test_id}/reports", exist_ok=True)
    os.makedirs(f"static/{user_test_id}/summary", exist_ok=True)
    bp_data = {
        '日期': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(35)],
        '早上收縮壓 (mmHg)': [120+i%10 - 5 for i in range(35)],
        '早上舒張壓 (mmHg)': [80+i%5 - 2 for i in range(35)],
        '早上脈搏 (次/分鐘)': [70 + i % 5 for i in range(35)],
        '中午收縮壓 (mmHg)': [125+i%10 - 5 for i in range(35)],
        '中午舒張壓 (mmHg)': [78+i%5 - 2 for i in range(35)],
        '中午脈搏 (次/分鐘)': [72 + i % 5 for i in range(35)],
        '晚上收縮壓 (mmHg)': [130+i%12 - 6 for i in range(35)],
        '晚上舒張壓 (mmHg)': [82+i%6 - 3 for i in range(35)],
        '晚上脈搏 (次/分鐘)': [68 + i % 5 for i in range(35)]
    }
    dummy_bp_df = pd.DataFrame(bp_data)
    dummy_bp_df.to_csv(dummy_data_path, index=False, encoding='utf-8-sig')
    print(f"\n--- Testing Blood Pressure Trend Analysis (7 days) ---")
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    bp_results = health_trend_analysis(dummy_data_path, user_test_id, ts, "7days")
    if bp_results.get("error"):
        print(f"Error: {bp_results['error']}")
    else:
        print(f"BP Preview Plot URL (relative to static): {bp_results['preview_plot_url']}")
        print(f"BP PDF Report URL (relative to static): {bp_results['pdf_report_url']}")
        print(f"BP Trend Analysis HTML snippet:\n{bp_results['trend_analysis_html'][:200]}...")
        if bp_results['preview_plot_url']: print(f"Preview plot created at: static/{bp_results['preview_plot_url']}")
        if bp_results['pdf_report_url']: print(f"PDF report created at: static/{bp_results['pdf_report_url']}")
    sugar_data = {
        '日期': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)],
        '早上空腹血糖 (mg/dL)': [90+i%8 - 4 for i in range(10)],
        '早上餐後血糖 (mg/dL)': [130+i%20 - 10 for i in range(10)],
        '中午空腹血糖 (mg/dL)': [95+i%8 - 4 for i in range(10)],
        '中午餐後血糖 (mg/dL)': [135+i%20 - 10 for i in range(10)],
        '晚餐前血糖 (mg/dL)': [92+i%8 - 4 for i in range(10)],
        '晚餐後血糖 (mg/dL)': [140+i%20 - 10 for i in range(10)]
    }
    dummy_sugar_df = pd.DataFrame(sugar_data)
    dummy_sugar_df.to_csv(dummy_data_path, index=False, encoding='utf-8-sig')
    print(f"\n--- Testing Blood Sugar Trend Analysis (today) ---")
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    sugar_results = health_trend_analysis(dummy_data_path, user_test_id, ts, "today")
    if sugar_results.get("error"):
        print(f"Error: {sugar_results['error']}")
    else:
        print(f"Sugar Preview Plot URL (relative to static): {sugar_results['preview_plot_url']}")
        print(f"Sugar PDF Report URL (relative to static): {sugar_results['pdf_report_url']}")
        print(f"Sugar Trend Analysis HTML snippet:\n{sugar_results['trend_analysis_html'][:200]}...")
        if sugar_results['preview_plot_url']: print(f"Preview plot created at: static/{sugar_results['preview_plot_url']}")
        if sugar_results['pdf_report_url']: print(f"PDF report created at: static/{sugar_results['pdf_report_url']}")
    print(f"\n--- Testing BP Summary Generation ---")
    try:
        summary_test_df = dummy_bp_df.head(5).copy()
        processed_summary_df = process_health_summary(summary_test_df, 'blood_pressure')
        if not processed_summary_df.empty:
            summary_html = generate_html_for_summary(processed_summary_df, title="血壓健康摘要")
            summary_pdf_path = generate_pdf_from_html_summary(summary_html, user_test_id, f"bp_summary_test_{ts}.pdf")
            print(f"BP Summary PDF generated at (relative to static): {summary_pdf_path}")
            print("Processed Summary DF:")
            print(processed_summary_df.head())
        else:
            print("Failed to generate processed summary DataFrame.")
    except Exception as e:
        print(f"Error during summary generation test: {e}")
    print(f"\nNote: Dummy CSV '{dummy_data_path}' may need to be manually removed.")