import os
import pandas as pd
import google.generativeai as genai
import pdfkit
from jinja2 import Template
from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from snownlp import SnowNLP

matplotlib.use('Agg')
matplotlib.rc('font', family='Microsoft JhengHei')

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Configure wkhtmltopdf
WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH")
if not WKHTMLTOPDF_PATH:
    raise EnvironmentError("WKHTMLTOPDF_PATH is not set or the file does not exist at the specified path.")
config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

# Prompts
default_prompt = """
你是一位長照輔助分析專家，請根據以下照護紀錄生成每日健康摘要，表格格式如下：

| 日期 | 日誌內容 | 食慾 | 睡眠狀況 | 行動能力 | 養護建議 |
|------|---------|------|---------|---------|----------|
"""

trend_prompt = """
你是一位健康數據分析師，請根據以下照護紀錄分析「異常趨勢」，例如：血壓連日上升、睡眠不足、活動量下降等，並給出建議。

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
    <h2>每日健康摘要</h2>
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

def generate_html(df: pd.DataFrame) -> str:
    template = Template(HTML_TEMPLATE)
    return template.render(table=df)

def generate_pdf_from_html(html_content: str) -> str:
    pdf_filename = "static/health_summary.pdf"
    os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)
    pdfkit.from_string(html_content, pdf_filename, configuration=config)
    return pdf_filename

def generate_mood_trend_plot(user_id, user_entries):
    output_dir = "static/moodtrend"
    os.makedirs(output_dir, exist_ok=True)

    # Convert date and sort
    user_entries["日期"] = pd.to_datetime(user_entries["日期"])
    user_entries = user_entries.sort_values("日期")
    # Convert mood index to numeric
    user_entries["心情指數"] = pd.to_numeric(user_entries.get("心情指數", 0), errors="coerce")
    # Perform sentiment analysis on notes
    user_entries["心情小語分析"] = user_entries["日誌內容"].apply(lambda text: SnowNLP(str(text)).sentiments * 9 + 1)

    # Calculate averages
    avg_recorded = user_entries["心情指數"].mean() if "心情指數" in user_entries else 0
    avg_snownlp = user_entries["心情小語分析"].mean()

    plt.figure(figsize=(12, 6))
    if "心情指數" in user_entries and user_entries["心情指數"].notna().any():
        sns.lineplot(x="日期", y="心情指數", data=user_entries, marker="o", label="用戶心情紀錄", color="blue", errorbar=None)
    sns.lineplot(x="日期", y="心情小語分析", data=user_entries, marker="o", label="SnowNLP 心情分析", color="red", errorbar=None)
    if avg_recorded:
        plt.axhline(y=avg_recorded, color='orange', linestyle='--', label=f"記錄平均 ({avg_recorded:.2f})")
    plt.axhline(y=avg_snownlp, color='green', linestyle='--', label=f"分析平均 ({avg_snownlp:.2f})")
    plt.xlabel("日期")
    plt.ylabel("心情指數")
    plt.title(f"用戶 {user_id} 的心情趨勢圖")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylim(1, 10)

    output_path = os.path.join(output_dir, f"mood_trend_{user_id}.png")
    plt.savefig(output_path)
    plt.close()

    return output_path

def process_health_summary(file_path, prompt):
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    df = pd.read_csv(file_path)
    block_size = 30
    cumulative_response = ""

    for i in range(0, df.shape[0], block_size):
        block = df.iloc[i:i+block_size]
        block_csv = block.to_csv(index=False)
        full_prompt = f"照護紀錄如下：\n{block_csv}\n\n{prompt}"
        response = model.generate_content(full_prompt)
        cumulative_response += response.text.strip() + "\n\n"

    df_result = parse_markdown_table(cumulative_response)
    if df_result is not None:
        html_content = generate_html(df_result)
        pdf_path = generate_pdf_from_html(html_content)
        return html_content, pdf_path
    else:
        return "⚠️ 無法解析 AI 回應內容", None

def health_trend_analysis(file_path):
    if not os.path.exists(file_path):
        return "請先上傳 CSV 檔案"

    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    df = pd.read_csv(file_path)
    user_id = os.path.splitext(os.path.basename(file_path))[0]
    
    # Generate mood trend plot
    plot_path = generate_mood_trend_plot(user_id, df)
    
    # Perform trend analysis
    content = df.to_csv(index=False)
    response = model.generate_content(f"{trend_prompt}\n\n{content}")
    trend_text = response.text.strip()
    
    # Combine results
    return f"{trend_text}\n\n📊 心情趨勢圖已生成：/static/moodtrend/mood_trend_{user_id}.png"

def answer_care_question(user_question):
    if not user_question.strip():
        return "請輸入問題"
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    prompt = rag_prompt_template.format(question=user_question.strip())
    response = model.generate_content(prompt)
    return response.text.strip()