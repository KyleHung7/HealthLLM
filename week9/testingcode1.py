import os
from datetime import datetime
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import pdfkit
from jinja2 import Template

# 設定 wkhtmltopdf 路徑
WKHTMLTOPDF_PATH = "D:/wkhtmltopdf/bin/wkhtmltopdf.exe"
config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

# 載入環境變數
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 預設 AI prompt
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

# HTML 模板
HTML_TEMPLATE = """
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid black; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
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

# 解析 Markdown 表格
def parse_markdown_table(markdown_text: str) -> pd.DataFrame:
    lines = [line.strip() for line in markdown_text.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if not table_lines or len(table_lines) < 3:
        return None
    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    data = [[cell.strip() for cell in line.strip("|").split("|")] for line in table_lines[2:]]
    return pd.DataFrame(data, columns=headers)

# 生成 HTML
def generate_html(df: pd.DataFrame) -> str:
    template = Template(HTML_TEMPLATE)
    return template.render(table=df)

# 轉成 PDF
def generate_pdf_from_html(html_content: str) -> str:
    pdf_filename = "health_summary.pdf"
    pdfkit.from_string(html_content, pdf_filename, configuration=config)
    return pdf_filename

# Step 1~3：處理健康摘要與 PDF
def process_health_summary(csv_file, prompt):
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    df = pd.read_csv(csv_file.name)
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

# Step 4：健康趨勢分析
def health_trend_analysis(csv_file):
    if csv_file is None:
        return "請先上傳 CSV 檔案"

    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    df = pd.read_csv(csv_file.name)
    content = df.to_csv(index=False)
    response = model.generate_content(f"{trend_prompt}\n\n{content}")
    return response.text.strip()

# Step 5：RAG 問答
def answer_care_question(user_question):
    if not user_question.strip():
        return "請輸入問題"
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    prompt = rag_prompt_template.format(question=user_question.strip())
    response = model.generate_content(prompt)
    return response.text.strip()

# Gradio 介面
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI 長照輔助系統（對應流程圖 Step 1–5）")

    with gr.Accordion("📝 Step 1~3：健康摘要與報告產生", open=True):
        csv_input = gr.File(label="📂 上傳照護紀錄（CSV）")
        user_input = gr.Textbox(label="分析指令", lines=8, value=default_prompt)
        submit_button = gr.Button("📄 產出健康摘要與 PDF")
        output_html = gr.HTML(label="HTML 預覽")
        output_pdf = gr.File(label="下載 PDF 報告")
        submit_button.click(fn=process_health_summary, inputs=[csv_input, user_input], outputs=[output_html, output_pdf])

    with gr.Accordion("📈 Step 4：健康趨勢分析", open=False):
        trend_button = gr.Button("🔍 分析健康趨勢與警示")
        trend_output = gr.Textbox(label="趨勢分析結果", lines=10)
        trend_button.click(fn=health_trend_analysis, inputs=[csv_input], outputs=trend_output)

    with gr.Accordion("💬 Step 5：照護問題即時問答（RAG）", open=False):
        care_question = gr.Textbox(label="輸入照護問題，例如：長輩失眠怎麼辦？", lines=2)
        care_button = gr.Button("💡 提出建議")
        care_output = gr.Textbox(label="AI 回覆", lines=10)
        care_button.click(fn=answer_care_question, inputs=care_question, outputs=care_output)

demo.launch()
