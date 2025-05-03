import os
import pandas as pd
import google.generativeai as genai
import pdfkit
from jinja2 import Template
from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from lib import mdToHtml

# Configure matplotlib
matplotlib.use('Agg')
matplotlib.rc('font', family='Microsoft JhengHei')

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Configure wkhtmltopdf
WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH")
if not WKHTMLTOPDF_PATH:
    raise EnvironmentError("WKHTMLTOPDF_PATH is not set or the file does not exist.")
config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

# Prompts
blood_pressure_prompt = """
你是一位長照輔助分析專家，請根據以下長者每日的血壓紀錄，提供簡潔的健康摘要與建議，並判斷是否達標。

請輸出下列表格格式：

| 日期 | 早上收縮壓 (mmHg) | 早上舒張壓 (mmHg) | 早上脈搏 (次/分鐘) | 晚上收縮壓 (mmHg) | 晚上舒張壓 (mmHg) | 晚上脈搏 (次/分鐘) | 達標狀況 | 養護建議 |
|------|-------------------|-------------------|---------------------|-------------------|-------------------|---------------------|-----------|----------|

請依據常見血壓標準（正常收縮壓 <130 且舒張壓 <80）判斷是否達標。
"""

blood_sugar_prompt = """
你是一位長照輔助分析專家，請根據以下長者每日的血糖紀錄，提供簡潔的健康摘要與建議，並判斷是否達標。

請輸出下列表格格式：

| 日期 | 早餐前血糖 | 早餐後2小時血糖 | 午餐前血糖 | 午餐後2小時血糖 | 晚餐前血糖 | 晚餐後2小時血糖 | 達標狀況 | 養護建議 |
|------|------------|------------------|------------|------------------|------------|------------------|-----------|----------|

請依據常見血糖標準（空腹血糖 <100 mg/dL，餐後兩小時 <140 mg/dL）判斷是否達標。
"""

trend_prompt = """
你是一位健康數據分析師，請根據以下血壓或血糖紀錄，分析是否出現異常趨勢（如連續升高、波動劇烈等），並提供簡短建議。

請輸出格式如下：
- 🟡 指標變化：...
- 🔴 建議：...
"""

rag_prompt_template = """
你是長照照護助手，請根據你掌握的知識，針對照顧者的問題提供具體建議。
問題：「{question}」
"""

# HTML Template
HTML_TEMPLATE = """
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid black; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 100%; margin-top: 20px; }
    </style>
</head>
<body>
    <h2>{{ title }}</h2>
    <table>
        <thead>
            <tr>
                {% for col in table.columns %}
                <th>{{ col }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in table.values %}
            <tr>
                {% for cell in row %}
                <td>{{ cell }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

def parse_markdown_table(markdown_text: str) -> pd.DataFrame:
    lines = [line.strip() for line in markdown_text.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if not table_lines or len(table_lines) < 3:
        return None
    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    data = [[cell.strip() for cell in line.strip("|").split("|")] for line in table_lines[2:]]
    return pd.DataFrame(data, columns=headers)

def generate_html(df: pd.DataFrame, title="健康紀錄分析") -> str:
    template = Template(HTML_TEMPLATE)
    return template.render(table=df, title=title)

def generate_pdf_from_html(html_content: str, pdf_filename: str) -> str:
    pdf_path = f"static/{pdf_filename}"
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    pdfkit.from_string(html_content, pdf_path, configuration=config)
    return pdf_path

def validate_bp_csv(df):
    required_columns = [
        '日期', '早上收縮壓 (mmHg)', '早上舒張壓 (mmHg)', '早上脈搏 (次/分鐘)',
        '晚上收縮壓 (mmHg)', '晚上舒張壓 (mmHg)', '晚上脈搏 (次/分鐘)'
    ]
    # 忽略大小寫和空格進行驗證
    df_columns = [col.strip().lower() for col in df.columns]
    required_columns = [col.strip().lower() for col in required_columns]
    return all(col in df_columns for col in required_columns)

def validate_sugar_csv(df):
    required_columns = [
        '日期', '早餐前血糖', '早餐後2小時血糖', '午餐前血糖',
        '午餐後2小時血糖', '晚餐前血糖', '晚餐後2小時血糖'
    ]
    # 忽略大小寫和空格進行驗證
    df_columns = [col.strip().lower() for col in df.columns]
    required_columns = [col.strip().lower() for col in required_columns]
    return all(col in df_columns for col in required_columns)

def process_health_summary(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    prompt = blood_pressure_prompt if data_type == 'blood_pressure' else blood_sugar_prompt
    df = df.fillna("無")
    content = df.to_csv(index=False)
    response = model.generate_content(f"{prompt}\n\n{content}")
    markdown = response.text.strip()

    summary_df = parse_markdown_table(markdown)
    if summary_df is None:
        raise ValueError("無法解析模型輸出的表格格式")
    return summary_df

def generate_health_trend_plot(file_path, output_file, columns, ylabel, title):
    try:
        df = pd.read_csv(file_path)
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values(by="日期")

        plt.figure(figsize=(12, 6))
        for col in columns:
            if col in df.columns:
                sns.lineplot(data=df, x="日期", y=col, label=col)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        return output_file
    except Exception as e:
        print(f"生成趨勢圖錯誤: {str(e)}")
        return None

def health_trend_analysis(file_path, user_id):
    if not os.path.exists(file_path):
        return "請先上傳 CSV 檔案"

    df = pd.read_csv(file_path)
    df.fillna("無", inplace=True)

    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    plot_path = None
    data_type = None

    try:
        if validate_bp_csv(df):
            data_type = 'blood_pressure'
            columns = ['早上收縮壓 (mmHg)', '早上舒張壓 (mmHg)', '晚上收縮壓 (mmHg)', '晚上舒張壓 (mmHg)']
            plot_path = generate_health_trend_plot(
                file_path,
                f"static/moodtrend/bp_trend_{user_id}.png",
                columns,
                "mmHg",
                "血壓趨勢圖"
            )
        elif validate_sugar_csv(df):
            data_type = 'blood_sugar'
            columns = ['早餐前血糖', '早餐後2小時血糖', '午餐前血糖', '午餐後2小時血糖', '晚餐前血糖', '晚餐後2小時血糖']
            plot_path = generate_health_trend_plot(
                file_path,
                f"static/moodtrend/sugar_trend_{user_id}.png",
                columns,
                "mg/dL",
                "血糖趨勢圖"
            )
        else:
            return "CSV 檔案格式不符合血壓或血糖分析要求"

        content = df.to_csv(index=False)
        response = model.generate_content(f"{trend_prompt}\n\n{content}")
        trend_text = response.text.strip()
        trend_html = mdToHtml(trend_text)

        if plot_path:
            return f"{trend_html}\n\n📊 {data_type}_trend 趨勢圖已生成<br><img style='width: 100%;' src='{plot_path}'/>"
        return f"{trend_text}\n\n⚠️ 未成功生成趨勢圖"
    except Exception as e:
        return f"趨勢分析錯誤: {str(e)}"

def answer_care_question(user_question):
    if not user_question.strip():
        return "請輸入問題"
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    prompt = rag_prompt_template.format(question=user_question.strip())
    response = model.generate_content(prompt)
    return response.text.strip()
