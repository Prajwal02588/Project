# SmartStudy AI 🎓

> **An agentic AI-powered study companion** built with Flask + Groq LLM. Upload your notes, PDFs, or images and instantly get structured summaries, smart flashcards, adaptive quizzes, personalized study plans, concept maps, and exam predictions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **AI Summarization** | Paste or upload content → get structured bullet-point summaries |
| 🃏 **Smart Flashcards** | Auto-generated Q&A cards with difficulty ratings + flip animation |
| ✏️ **Adaptive Quizzes** | MCQ & short-answer quizzes with timer, scoring, and AI analysis |
| 📅 **Study Plans** | Day-by-day personalized schedules based on your topics & deadline |
| 🔮 **Exam Predictions** | AI predicts high-probability exam topics with confidence scores |
| 🗺️ **Concept Maps** | Hierarchical visual maps from study material |
| 💡 **Concept Explainer** | Deep explanations with examples, analogies & memory tips |
| 📊 **Progress Dashboard** | Quiz scores, weak areas, achievements, AI recommendations |
| 💬 **Agentic Chat** | ReAct-loop agent that picks the right tool for any study request |
| 📂 **File Extraction** | PDF, TXT, PNG/JPG (OCR), WAV/MP3 (speech-to-text) |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/SmartStudyApp.git
cd SmartStudyApp
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
```bash
cp .env.example .env
```
Edit `.env` and add your **free** Groq API key (get one at [console.groq.com](https://console.groq.com)):
```env
GROQ_API_KEY=gsk_your_key_here
```

### 5. Run
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🏗️ Project Structure

```
SmartStudyApp/
├── app.py              # Flask application & API routes
├── agent.py            # Agentic AI core (ReAct loop, tool calling)
├── tools.py            # Study tool functions (summarize, quiz, etc.)
├── config.py           # Configuration & environment variables
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── static/
│   ├── css/style.css   # Full design system
│   └── js/main.js      # Shared JS utilities
├── templates/
│   ├── base.html       # Base layout
│   ├── index.html      # Home page + chat
│   ├── study.html      # Study tools page
│   ├── quiz.html       # Interactive quiz page
│   └── dashboard.html  # Progress dashboard
├── uploads/            # Uploaded files (gitignored)
└── data/               # Progress JSON files (gitignored)
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `SECRET_KEY` | `smart-study-secret-2024` | Flask session secret |
| `DEBUG` | `True` | Flask debug mode |

**Available models (set via `GROQ_MODEL` in `.env`):**
- `llama-3.3-70b-versatile` — Best quality, tool_use supported *(default)*
- `llama-3.1-8b-instant` — Fastest / lowest latency
- `llama3-70b-8192` — Legacy 70B
- `mixtral-8x7b-32768` — Long context (no tool_use)

---

## 🛠️ Tech Stack

- **Backend:** Python 3.11, Flask 3.0, Groq SDK 1.7
- **LLM:** Groq Cloud — llama-3.3-70b-versatile
- **Frontend:** Vanilla HTML/CSS/JS (zero frameworks)
- **File processing:** PyPDF2, Pillow + pytesseract, SpeechRecognition
- **Design:** Dark glassmorphism UI with CSS custom properties

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Agentic chat message |
| `POST` | `/api/chat/reset` | Clear conversation history |
| `POST` | `/api/upload` | Upload & extract file text |
| `POST` | `/api/process` | Run study action (summarize/flashcards/quiz/etc.) |
| `POST` | `/api/study-plan` | Generate a study plan |
| `POST` | `/api/predict-topics` | Predict exam topics |
| `POST` | `/api/submit-quiz` | Submit quiz answers & get analysis |
| `POST` | `/api/explain` | Explain a concept |
| `GET`  | `/api/progress` | Get session progress data |
| `POST` | `/api/progress/update` | Update progress data |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ using <a href="https://groq.com">Groq</a> + Flask</p>
