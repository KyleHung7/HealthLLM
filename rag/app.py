from flask import Flask, request, jsonify, send_file
import tempfile
from gtts import gTTS
import whisper
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from pydub import AudioSegment
import asyncio
import uuid
import hashlib
from tqdm import tqdm
import torch
from importlib.util import find_spec

load_dotenv()

if find_spec("intel_extension_for_pytorch") is not None:
    import intel_extension_for_pytorch

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Temporary workaround for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.info("Set KMP_DUPLICATE_LIB_OK=TRUE to handle OpenMP conflict (temporary workaround).")

# Specify FFmpeg path
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "C:\\ffmpeg\\bin")  # Default path if not set
if os.path.exists(os.path.join(FFMPEG_PATH, "ffmpeg.exe")) and os.path.exists(os.path.join(FFMPEG_PATH, "ffprobe.exe")):
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH
    AudioSegment.ffmpeg = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")
    logging.info(f"FFmpeg path added to environment: {FFMPEG_PATH}")
else:
    logging.error(f"FFmpeg executables not found in {FFMPEG_PATH}. Please ensure ffmpeg.exe and ffprobe.exe are present.")
    raise FileNotFoundError("FFmpeg executables not found.")

# Initialize Whisper with GPU/CPU detection
try:
    device = None;
    if torch.cuda.is_available():
        device = "cuda"
    elif find_spec('torch.xpu') is not None and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    logging.info(f"Using device: {device}")
    whisper_model = whisper.load_model("base", device=device)
    logging.info(f"Whisper model loaded on {device}.")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")
    raise

# Initialize FAISS with PDF hash check
PDF_DATA_DIR = os.getenv("PDF_DATA_DIR")
index_path = "faiss_index"
print(PDF_DATA_DIR, os.listdir(PDF_DATA_DIR))
pdf_paths = []
if PDF_DATA_DIR:
    for filename in os.listdir(PDF_DATA_DIR):
        if filename.endswith(".pdf"):
            pdf_paths.append(os.path.join(PDF_DATA_DIR, filename))
else:
    logging.warning("PDF_DATA_DIR environment variable not set.")
hash_file = "pdf_hash.txt"
model_name = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

def get_pdf_hash(paths):
    hasher = hashlib.sha256()
    for path in sorted(paths):
        try:
            with open(path, 'rb') as f:
                hasher.update(f.read())
        except FileNotFoundError:
            logging.warning(f"PDF file not found: {path}")
            continue
    return hasher.hexdigest()

try:
    embeddings = OllamaEmbeddings(model=model_name)
    logging.info("Ollama embeddings initialized successfully.")
except Exception as e:
    logging.error(f"Ollama embeddings initialization failed ({model_name}): {e}")
    raise

vectorstore = None
current_pdf_hash = get_pdf_hash(pdf_paths)

if os.path.exists(index_path) and os.path.exists(hash_file):
    try:
        with open(hash_file, 'r') as f:
            saved_hash = f.read().strip()
        if saved_hash == current_pdf_hash:
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            logging.info("Loaded existing FAISS index.")
        else:
            logging.info("PDF files have changed, regenerating FAISS index.")
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}")

if not vectorstore:
    documents = []
    for path in tqdm(pdf_paths, desc="Loading PDF files"):
        try:
            loader = PyPDFLoader(path)
            for page in tqdm(loader.load(), desc=f"Processing {os.path.basename(path)}", leave=False):
                documents.append(page)
        except FileNotFoundError:
            logging.warning(f"File not found: {path}")
            continue
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            continue

    if not documents:
        raise ValueError("No documents loaded, please check PDF paths or formats.")

    texts = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150
    ).split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)

    try:
        vectorstore.save_local(index_path)
        with open(hash_file, 'w') as f:
            f.write(current_pdf_hash)
        logging.info(f"FAISS index saved to {index_path}, hash saved to {hash_file}")
    except Exception as e:
        logging.error(f"Failed to save FAISS index or hash: {e}")
        raise

# Set up retriever
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    raise ValueError("GEMINI_API_KEY is required.")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")
logging.info("Gemini model initialized successfully.")

async def transcribe_audio(audio_path):
    try:
        return await asyncio.to_thread(whisper_model.transcribe, audio_path)
    except Exception as e:
        logging.error(f"Whisper transcription failed for {audio_path}: {e}")
        raise

async def process_question(question):
    try:
        retriever_chain = RetrievalQA.from_chain_type(
            llm=OllamaLLM(model=model_name, system="你是一個專業的助手，所有回應請使用正體中文，語言清晰且符合台灣用語習慣。"),
            retriever=retriever
        )
        retrieval_response = retriever_chain.invoke(question)
        context = retrieval_response["result"] if isinstance(retrieval_response, dict) else retrieval_response
        logging.info("LLaMA retrieval successful.")

        final_prompt = f"""你是一個健康知識助手，專注於血壓和血糖管理，請根據下列檢索到的資訊與問題，提供清楚、符合台灣用語的專業回答，特別考慮長者需求。

❗ 問題：
{question}

🔍 檢索結果：
{context}

請注意：你的回覆應該使用繁體中文，簡潔明瞭，適合長者理解。
"""
        response = gemini.generate_content(final_prompt)
        answer = response.text.strip()
        logging.info(f"Gemini response: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Question processing failed: {e}")
        return f"錯誤：{str(e)}"

AUDIO_OUTPUT_DIR = os.getenv("AUDIO_OUTPUT_DIR")

async def generate_speech(text):
    try:
        tts = gTTS(text=text, lang='zh-tw')
        tts_path = os.path.join(AUDIO_OUTPUT_DIR, f"response_{uuid.uuid4()}.mp3")
        tts.save(tts_path)
        logging.info(f"Generated audio response at: {tts_path}")
        return tts_path
    except Exception as e:
        logging.error(f"Speech generation failed: {e}")
        return None

@app.route('/record', methods=['POST'])
async def record():
    try:
        mode = request.form.get('mode')  # 'transcribe' or 'voice'
        if 'audio' not in request.files:
            logging.error("No audio file provided.")
            return jsonify({"error": "No audio file provided."}), 400

        audio_file = request.files['audio']
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"recording_{uuid.uuid4()}{os.path.splitext(audio_file.filename)[1]}")
        audio_file.save(audio_path)
        logging.info(f"Saved audio to: {audio_path}")

        try:
            # Validate and convert audio file
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path if audio_path.endswith('.wav') else os.path.join(temp_dir, f"converted_{uuid.uuid4()}.wav")
            audio.export(wav_path, format="wav")
            normalized_path = Path(wav_path).as_posix()
            logging.info(f"Processed audio to: {normalized_path}")
        except Exception as e:
            logging.error(f"Audio processing failed: {e}")
            logging.info(f"Attempting to transcribe raw audio file: {audio_path}")
            normalized_path = Path(audio_path).as_posix()  # Fallback to raw file

        try:
            result = await transcribe_audio(normalized_path)
            question = result["text"].strip()
            logging.info(f"Transcribed text: {question}")
        except Exception as e:
            logging.error(f"Whisper transcription failed: {e}")
            return jsonify({"error": f"語音轉錄失敗：{str(e)}"}), 500

        try:
            os.remove(audio_path)
            if wav_path != audio_path and os.path.exists(wav_path):
                os.remove(wav_path)
            logging.debug(f"Deleted temporary files: {audio_path}, {wav_path}")
        except Exception as e:
            logging.warning(f"Failed to delete temporary files: {e}")

        if mode == 'transcribe':
            return jsonify({"transcription": question})
        elif mode == 'voice':
            answer = await process_question(question)
            tts_path = await generate_speech(answer)
            if tts_path:
                return jsonify({"transcription": question, "answer": answer, "audio": f"/audio/{os.path.basename(tts_path)}"})
            return jsonify({"transcription": question, "answer": answer})
        else:
            return jsonify({"error": "Invalid mode."}), 400

    except Exception as e:
        logging.error(f"Recording failed: {e}")
        return jsonify({"error": f"處理失敗：{str(e)}"}), 500

@app.route('/submit', methods=['POST'])
async def submit():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            logging.error("No question provided in JSON body.")
            return jsonify({"error": "請輸入問題。"}), 400
        question = data.get('question')

        answer = await process_question(question)
        voice_mode = data.get('voice_mode', False) # Default to False if not provided
        if voice_mode:
            tts_path = await generate_speech(answer)
            if tts_path:
                return jsonify({"answer": answer, "audio": f"/audio/{os.path.basename(tts_path)}"})
        return jsonify({"answer": answer})

    except Exception as e:
        logging.error(f"Submission failed: {e}")
        return jsonify({"error": f"處理失敗：{str(e)}"}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/mpeg')
    return jsonify({"error": "Audio file not found."}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)