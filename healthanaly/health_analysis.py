import os
import pandas as pd
import google.generativeai as genai
import pdfkit
from jinja2 import Template
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta
import io
import base64
import markdown
import re 
import json

# 導入 Plotly Express 和 Graph Objects
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Import for dual y-axis
import plotly.io as pio

# 設定 Plotly 預設主題
pio.templates.default = "plotly_white"

# 載入環境變數
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-1.5-flash")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("警告：未找到 GEMINI_API_KEY。AI 生成功能將會失敗。")

# 設定 wkhtmltopdf 路徑
WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH")
config = None
if WKHTMLTOPDF_PATH and os.path.exists(WKHTMLTOPDF_PATH):
    config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
else:
    print("警告：未設定 WKHTMLTOPDF_PATH 或執行檔不存在。PDF 生成功能將會失敗。")

# --- 提示詞 (Prompts) ---
trend_prompt = """
你是一位專業的健康數據分析師。請根據以下提供的健康數據紀錄（已根據用戶選擇的時間區間篩選），分析數據中是否存在任何顯著的異常趨勢（例如，指標持續升高、持續降低、波動過於劇烈、頻繁超出正常範圍等）。
請提供簡短的觀察結果和針對這些趨勢的初步建議。

請嚴格按照以下格式輸出，使用 Markdown 語法：
- 🟡 **指標變化觀察**：[在這裡描述你觀察到的數據模式，例如：過去一週血壓有輕微上升趨勢，尤其是在晚間。]

- 🔴 **健康建議**：
  - [根據觀察到的趨勢提供第一條具體建議，例如：建議持續監測血壓，並記錄每日活動和飲食，以找出潛在影響因素。]
  - [提供第二條具體建議，例如：考慮調整晚間作息，確保充足睡眠，避免睡前攝入咖啡因或高糖食物。]
  - [提供第三條具體建議，例如：若血壓持續升高或出現不適，請務必諮詢醫療專業人員，切勿自行調整藥物。]
"""

# --- PDF 報告的 HTML 模板 ---
PDF_REPORT_TEMPLATE = """
<html>
<head>
    <meta charset="utf-8">
    <title>{{ report_title }}</title>
    <style>
        /* 為了簡潔，這裡省略了 Base64 字體，實際部署時應包含 */
        body { 
            font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif; 
            margin: 20px; 
            color: #333; 
        }
        h1, h2, h3 { color: #2c7a7b; }
        h1 { text-align: center; margin-bottom: 30px; }
        h2 { border-bottom: 2px solid #a3d5d3; padding-bottom: 5px; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; word-wrap: break-word; }
        th { background-color: #def7f6; font-weight: bold; }
        img { max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; padding: 5px; }
        .analysis-section { margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }
        .analysis-section p { white-space: pre-wrap; line-height: 1.6; }
        .analysis-section ul { margin-left: 20px; padding-left: 0; list-style-type: disc; }
        .analysis-section li { margin-bottom: 5px; }
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

    {% if trend_plot_html_tag %}
    <h2>趨勢圖</h2>
    {{ trend_plot_html_tag | safe }}
    {% endif %}

    {% if trend_analysis_text %}
    <h2>趨勢分析與建議 (AI生成)</h2>
    <div class="analysis-section">
        {{ trend_analysis_text | markdown_to_html | safe }}
    </div>
    {% endif %}

    <div class="footer">
        <p>HealthLLM 健康報告 - 生成時間: {{ generation_timestamp }}</p>
    </div>
</body>
</html>
"""

# Jinja2 過濾器
def markdown_to_html_filter(text):
    return markdown.markdown(text)

# --- 核心分析函式 ---

def analyze_blood_pressure(systolic, diastolic, pulse=None):
    if not (isinstance(systolic, (int, float)) and isinstance(diastolic, (int, float))):
        return "血壓輸入無效", "請輸入有效的數字作為血壓值。", ""
    if pulse is not None and not isinstance(pulse, (int, float)):
        return "脈搏輸入無效", "請輸入有效的數字作為脈搏值，或留空。", ""
    s = int(systolic)
    d = int(diastolic)
    status = "未知血壓狀態"
    advice = "請諮詢醫生以獲得專業評估。"
    normal_range_info = f"理想血壓: 收縮壓 < 120 mmHg 且 舒張壓 < 80 mmHg。"
    if s >= 180 or d >= 120:
        status = "高血壓危機"
        advice = "您的血壓非常高，這可能表示高血壓危機。請立即尋求醫療協助！"
    elif s >= 140 or d >= 90:
        status = "第二期高血壓"
        advice = "您的血壓處於第二期高血壓範圍，建議立即諮詢醫生，可能需要藥物治療和生活方式調整。"
    elif (130 <= s <= 139) or (80 <= d <= 89):
        status = "第一期高血壓"
        advice = "您的血壓處於第一期高血壓範圍，建議諮詢醫生討論生活方式改變，並定期監測。"
    elif (120 <= s <= 129) and d < 80:
        status = "血壓偏高"
        advice = "您的血壓略高於理想範圍，建議開始注意健康生活方式，如健康飲食、規律運動和減輕壓力。"
    elif s < 120 and d < 80:
        if s < 90 or d < 60:
             status = "血壓偏低"
             advice = "您的血壓在正常範圍內但偏低，如果伴有頭暈、乏力等症狀，請諮詢醫生。"
             normal_range_info = f"一般低血壓參考: 收縮壓 < 90 mmHg 或 舒張壓 < 60 mmHg。"
        else:
            status = "正常血壓"
            advice = "您的血壓在理想範圍，請繼續保持健康的生活習慣。"
    else:
        status = "血壓數據組合特殊"
        advice = "您的血壓數據組合較為特殊或不完整，建議諮詢醫生。"
    pulse_info = ""
    if pulse is not None:
        p = int(pulse)
        pulse_info = f"脈搏: {p} 次/分。 "
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
        normal_range_info = f"理想空腹血糖: 70-99 mg/dL"
        if val <= 69:
            status = "低血糖 (空腹)"
            advice = f"您的空腹血糖 ({val} mg/dL) 偏低，可能為低血糖。若有不適請立即補充糖分並諮詢醫生。"
        elif val <= 99:
            status = "正常空腹血糖"
            advice = f"您的空腹血糖 ({val} mg/dL) 在理想範圍。"
        elif 100 <= val <= 125:
            status = "糖尿病前期 (空腹)"
            advice = f"您的空腹血糖 ({val} mg/dL) 偏高，屬於糖尿病前期。建議改善飲食、增加運動，並定期追蹤血糖。"
        elif val >= 126:
            status = "糖尿病 (空腹)"
            advice = f"您的空腹血糖 ({val} mg/dL) 明顯偏高，可能已達糖尿病標準。請立即諮詢醫生進行進一步檢查和治療。"
    elif measurement_type.lower() == "postprandial":
        normal_range_info = f"理想餐後血糖 (餐後2小時): < 140 mg/dL"
        if val <= 69:
            status = "低血糖 (餐後)"
            advice = f"您的餐後血糖 ({val} mg/dL) 偏低，可能為低血糖。若有不適請立即補充糖分並諮詢醫生。"
        elif val <= 139:
            status = "正常餐後血糖"
            advice = f"您的餐後血糖 ({val} mg/dL) 在理想範圍。"
        elif 140 <= val <= 199:
            status = "糖尿病前期 (餐後)"
            advice = f"您的餐後血糖 ({val} mg/dL) 偏高，屬於糖尿病前期。建議改善飲食、增加運動，並定期追蹤血糖。"
        elif val >= 200:
            status = "糖尿病 (餐後)"
            advice = f"您的餐後血糖 ({val} mg/dL) 明顯偏高，可能已達糖尿病標準。請立即諮詢醫生進行進一步檢查和治療。"
    else:
        return "未知的血糖測量類型", "請指定 'fasting' (空腹) 或 'postprandial' (餐後)。", ""
    return status, advice, normal_range_info

def filter_data_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if 'Date' not in df.columns:
        raise ValueError("DataFrame 必須包含 'Date' 欄位。")
    df_filtered = df.copy()
    try:
        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['Date'])
    except Exception as e:
        raise ValueError(f"轉換 'Date' 欄位為日期格式時出錯: {e}")
    if df_filtered.empty:
        return df_filtered
    df_filtered = df_filtered.sort_values(by="Date", ascending=True)
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
        print(f"無法識別的時間範圍 '{period}'，將回傳所有數據。")
        return df_filtered

def reshape_for_plotting(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Reshapes the data from wide to long format suitable for plotting."""
    df_long = df.melt(id_vars=['Date'], var_name='Metric', value_name='Value')
    df_long.dropna(subset=['Value'], inplace=True)
    if df_long.empty:
        return pd.DataFrame()

    pattern = re.compile(r"(Morning|Noon|Evening)_(\w+)")
    
    extracted_data = df_long['Metric'].str.extract(pattern)
    df_long['TimeOfDayRaw'] = extracted_data[0]
    df_long['MetricTypeRaw'] = extracted_data[1]

    df_long.dropna(subset=['TimeOfDayRaw', 'MetricTypeRaw'], inplace=True)
    if df_long.empty:
        return pd.DataFrame()

    time_map = {'Morning': '早上', 'Noon': '中午', 'Evening': '晚上'}
    
    if data_type == 'blood_pressure':
        metric_map = {'Systolic': '收縮壓', 'Diastolic': '舒張壓', 'Pulse': '脈搏'}
    else: # blood_sugar
        metric_map = {'Fasting': '空腹血糖', 'Postprandial': '餐後血糖'}

    df_long['TimeOfDay'] = df_long['TimeOfDayRaw'].map(time_map)
    df_long['MetricType'] = df_long['MetricTypeRaw'].map(metric_map)
    
    df_long.dropna(subset=['TimeOfDay', 'MetricType'], inplace=True)
    
    time_offsets = {'早上': 8, '中午': 12, '晚上': 20}
    
    df_long['DateTime'] = df_long.apply(
        lambda row: row['Date'] + pd.to_timedelta(time_offsets.get(row['TimeOfDay'], 12), unit='h'),
        axis=1
    )
    
    if data_type == 'blood_sugar':
        postprandial_mask = df_long['MetricTypeRaw'] == 'Postprandial'
        df_long.loc[postprandial_mask, 'DateTime'] += pd.to_timedelta(2, unit='h')

    df_long.sort_values('DateTime', inplace=True)
    return df_long

def generate_plotly_data(reshaped_df: pd.DataFrame, data_type: str, title: str):
    if reshaped_df.empty:
        return "{}"

    reshaped_df['Value'] = pd.to_numeric(reshaped_df['Value'], errors='coerce')
    reshaped_df.dropna(subset=['Value'], inplace=True)

    if reshaped_df.empty:
        return "{}"

    metric_types = reshaped_df['MetricType'].unique()

    if data_type == 'blood_pressure':
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for metric in metric_types:
            df_metric = reshaped_df[reshaped_df['MetricType'] == metric]
            is_secondary = metric == '脈搏'
            
            fig.add_trace(
                go.Scatter(
                    x=df_metric['DateTime'].tolist(),
                    y=df_metric['Value'].tolist(),
                    mode='lines+markers',
                    name=metric,
                    hovertemplate=
                    '<b>%{y}</b><br>' +
                    '日期: %{x|%Y-%m-%d}<br>' +
                    '時間: %{x|%H:%M}<extra></extra>'
                ),
                secondary_y=is_secondary
            )
        
        fig.update_yaxes(title_text="<b>血壓</b> (mmHg)", range=[0, 300], secondary_y=False)
        fig.update_yaxes(title_text="<b>脈搏</b> (次/分)", range=[0, 200], secondary_y=True)

    else: # blood_sugar
        fig = go.Figure()
        for metric in metric_types:
            df_metric = reshaped_df[reshaped_df['MetricType'] == metric]
            fig.add_trace(
                go.Scatter(
                    x=df_metric['DateTime'].tolist(),
                    y=df_metric['Value'].tolist(),
                    mode='lines+markers',
                    name=metric,
                    hovertemplate=
                    '<b>%{y}</b><br>' +
                    '日期: %{x|%Y-%m-%d}<br>' +
                    '時間: %{x|%H:%M}<extra></extra>'
                )
            )
        fig.update_yaxes(title_text="<b>血糖</b> (mg/dL)", range=[0, 300])

    fig.update_layout(
        title={'text': title, 'font': {'size': 20}},
        xaxis_title='日期與時間',
        hovermode='x unified',
        legend_title_text='指標'
    )
    
    return fig.to_json()

def generate_plot_base64_with_plotly(plotly_fig_json_str: str):
    """Generates a base64 PNG from a Plotly figure JSON string."""
    if not plotly_fig_json_str or plotly_fig_json_str == "{}":
        return None
    try:
        fig = go.Figure(json.loads(plotly_fig_json_str))
        img_bytes = pio.to_image(fig, format='png', width=900, height=500, scale=2)
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"使用 Plotly 生成圖片時發生錯誤: {e}")
        return None

def generate_trend_report_pdf(
    base_output_dir: str, request_timestamp_str: str, report_title: str, 
    time_period_label: str, time_period_key: str,
    data_table_df: pd.DataFrame, 
    trend_plot_base64_data: str, trend_analysis_text: str,
    data_type_for_filename: str
) -> tuple[str | None, str | None]:
    if not config:
        print("錯誤：未設定 wkhtmltopdf。無法生成 PDF 報告。")
        return None, None

    # *** THE FIX IS HERE ***
    # Define the translation dictionary
    column_translation = {
        'Date': '日期',
        'Morning_Systolic': '早上收縮壓',
        'Morning_Diastolic': '早上舒張壓',
        'Morning_Pulse': '早上脈搏',
        'Noon_Systolic': '中午收縮壓',
        'Noon_Diastolic': '中午舒張壓',
        'Noon_Pulse': '中午脈搏',
        'Evening_Systolic': '晚上收縮壓',
        'Evening_Diastolic': '晚上舒張壓',
        'Evening_Pulse': '晚上脈搏',
        'Morning_Fasting': '早晨空腹血糖',
        'Morning_Postprandial': '早晨餐後血糖',
        'Noon_Fasting': '午間空腹血糖',
        'Noon_Postprandial': '午間餐後血糖',
        'Evening_Fasting': '晚間空腹血糖',
        'Evening_Postprandial': '晚間餐後血糖'
    }

    # Create a copy for display and rename columns
    data_table_df_display = data_table_df.copy()
    data_table_df_display.rename(columns=column_translation, inplace=True)
    
    # Fill NaN values and convert to HTML
    data_table_df_display = data_table_df_display.fillna('')
    data_table_html = data_table_df_display.to_html(index=False, border=0, classes="dataframe") if not data_table_df_display.empty else "<p>此期間無數據可顯示。</p>"
    
    generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template = Template(PDF_REPORT_TEMPLATE)
    template.environment.filters['markdown_to_html'] = markdown_to_html_filter

    trend_plot_html_tag = f'<img src="data:image/png;base64,{trend_plot_base64_data}" alt="趨勢圖">' if trend_plot_base64_data else ''

    html_content = template.render(
        report_title=report_title,
        time_period_label=time_period_label,
        data_table_html=data_table_html,
        trend_plot_html_tag=trend_plot_html_tag,
        trend_analysis_text=trend_analysis_text,
        generation_timestamp=generation_timestamp
    )
    
    report_type_str = "血壓" if data_type_for_filename == 'blood_pressure' else "血糖"
    
    period_days = ''.join(filter(str.isdigit, time_period_key))
    if not period_days:
        if time_period_key == 'today':
            period_days = '1'
        else:
            period_days = '所有'
            
    today_str = datetime.now().strftime("%Y-%m-%d")
    pdf_filename = f"{report_type_str}_{period_days}天_{today_str}.pdf"
    
    pdf_output_folder = os.path.join(base_output_dir, "reports")
    os.makedirs(pdf_output_folder, exist_ok=True)
    abs_pdf_path = os.path.abspath(os.path.join(pdf_output_folder, pdf_filename))
    
    try:
        pdfkit.from_string(html_content, abs_pdf_path, configuration=config, options={'enable-local-file-access': ''})
        
        static_dir_abs = os.path.abspath("static")
        relative_path = os.path.relpath(abs_pdf_path, static_dir_abs)
        url_compatible_path = relative_path.replace(os.path.sep, '/')
        
        return url_compatible_path, pdf_filename
        
    except Exception as e:
        print(f"生成 PDF 報告 '{pdf_filename}' 時發生錯誤: {e}")
        debug_html_path = os.path.join(os.path.dirname(abs_pdf_path), f"debug_trend_report_{request_timestamp_str}.html")
        with open(debug_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"除錯用的 HTML 已儲存至: {debug_html_path}")
        raise

def health_trend_analysis(csv_file_path: str, base_output_dir: str, analysis_timestamp_str: str, time_period_filter: str, data_type: str, generate_pdf: bool = True):
    try:
        if not os.path.exists(csv_file_path):
            return "錯誤：數據檔案不存在。", None, None, None

        df_original = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        
        if 'Date' not in df_original.columns:
            raise ValueError("CSV 檔案中缺少 'Date' 欄位。")

        bp_cols = [
            'Morning_Systolic', 'Morning_Diastolic', 'Morning_Pulse',
            'Noon_Systolic', 'Noon_Diastolic', 'Noon_Pulse',
            'Evening_Systolic', 'Evening_Diastolic', 'Evening_Pulse'
        ]
        sugar_cols = [
            'Morning_Fasting', 'Morning_Postprandial',
            'Noon_Fasting', 'Noon_Postprandial',
            'Evening_Fasting', 'Evening_Postprandial'
        ]

        cols_for_type = bp_cols if data_type == 'blood_pressure' else sugar_cols

        df_analysis = df_original.copy()
        for col in cols_for_type:
            if col in df_analysis.columns:
                df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        
        df_filtered = filter_data_by_period(df_analysis, time_period_filter)

        if df_filtered.empty:
            return "選定時間範圍內無數據可供分析。", None, "{}", None

        df_reshaped = reshape_for_plotting(df_filtered, data_type)

        if df_reshaped.empty:
             return "選定時間範圍內無有效數據可供分析。", None, "{}", None

        time_period_labels = {'today': '當日', '7days': '最近7天', '30days': '最近30天', 'all': '所有歷史數據'}
        time_label = time_period_labels.get(time_period_filter, "指定期間")
        
        report_title_str = "血糖趨勢分析報告" if data_type == 'blood_sugar' else "血壓與脈搏趨勢分析報告"
        
        plotly_data_string = generate_plotly_data(df_reshaped, data_type, f"{report_title_str} ({time_label})")

        trend_analysis_output_text = "AI趨勢分析未能生成。"
        if api_key:
            try:
                cols_to_show = ['Date'] + [col for col in cols_for_type if col in df_filtered.columns and not df_filtered[col].isnull().all()]
                prompt_df_display = df_filtered[cols_to_show].copy()
                prompt_df_display['Date'] = prompt_df_display['Date'].dt.strftime('%Y-%m-%d')
                data_for_prompt = prompt_df_display.to_string(index=False, na_rep='無')
                
                model = genai.GenerativeModel(gemini_model)
                full_trend_prompt = f"{trend_prompt}\n以下是分析數據 ({time_label}):\n{data_for_prompt}"
                response = model.generate_content(full_trend_prompt)
                trend_analysis_output_text = response.text.strip()
            except Exception as gemini_err:
                trend_analysis_output_text = f"AI趨勢分析時發生錯誤: {gemini_err}"
        else:
            trend_analysis_output_text = "AI模型API金鑰未設定，無法執行AI趨勢分析。"

        pdf_report_rel_static_path = None
        pdf_filename = None
        if generate_pdf:
            trend_plot_base64 = generate_plot_base64_with_plotly(plotly_data_string)
            
            pdf_table_cols = ['Date'] + [col for col in cols_for_type if col in df_filtered.columns and not df_filtered[col].isnull().all()]
            pdf_table_df_display = df_filtered[pdf_table_cols].copy()
            pdf_table_df_display['Date'] = pdf_table_df_display['Date'].dt.strftime('%Y-%m-%d')
            
            if config and trend_plot_base64:
                pdf_report_rel_static_path, pdf_filename = generate_trend_report_pdf(
                    base_output_dir=base_output_dir, request_timestamp_str=analysis_timestamp_str,
                    report_title=report_title_str, 
                    time_period_label=time_label,
                    time_period_key=time_period_filter,
                    data_table_df=pdf_table_df_display, 
                    trend_plot_base64_data=trend_plot_base64,
                    trend_analysis_text=trend_analysis_output_text,
                    data_type_for_filename=data_type 
                )
            else:
                trend_analysis_output_text += "\n(PDF報告生成已跳過，因系統未配置PDF引擎或趨勢圖生成失敗)"
        
        return trend_analysis_output_text, pdf_report_rel_static_path, plotly_data_string, pdf_filename

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"趨勢分析主流程發生錯誤: {str(e)}", None, None, None
