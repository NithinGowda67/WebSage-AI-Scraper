# WEBSAGE 🚀

An **AI-powered web scraping and semantic analysis platform** built with **Python, Streamlit, Gemini embeddings, and FAISS**.

WEBSAGE helps users **extract, analyze, search, and visualize web data intelligently** across multiple domains such as **e-commerce, jobs, travel, and real estate**.

---

## ✨ Features

* 🔎 **Smart Web Scraping** using BeautifulSoup
* 🧠 **Semantic Search** with Gemini embeddings
* ⚡ **Vector Similarity Search** powered by FAISS
* 📊 **Interactive Analytics Dashboard** using Streamlit
* 🌐 Multi-domain scraping support:

  * E-commerce products
  * Job listings
  * Travel data
  * Real estate listings
* 📁 **HTML Report Export** for analysis results
* 🔐 Basic login/user session support
* 🖥️ Windows executable packaging support

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **Web Scraping:** BeautifulSoup, Requests
* **AI Embeddings:** Gemini API
* **Vector Database:** FAISS
* **Data Processing:** Pandas, NumPy
* **Storage:** JSON / Local files

---

## 📂 Project Structure

```text
WEBSAGE/
│── app.py
│── faiss_index/
│── .streamlit/
│── analysis_results.html
│── users.json
│── requirements.txt
│── README.md
```

---

## 🚀 Installation & Setup

### 1) Clone the repository

```bash
git clone https://github.com/NithinGowda67/WEBSAGE.git
cd WEBSAGE
```

### 2) Create virtual environment

```bash
python -m venv venv
```

### 3) Activate environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 4) Install dependencies

```bash
pip install -r requirements.txt
```

### 5) Add environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

### 6) Run the project

```bash
streamlit run app.py
```

---

## 🎯 Use Cases

* Product price comparison
* Job market intelligence
* Travel data insights
* Property listing analysis
* AI-powered semantic search over scraped content

---

## 🔮 Future Enhancements

* 🌍 Live multi-page crawling
* ☁️ Cloud deployment (AWS/GCP)
* 👥 Multi-user authentication
* 📈 Advanced ML insights
* 🗂️ PostgreSQL / MongoDB storage
* 🤖 Chat-based search assistant

---

## 👨‍💻 Author

**Nithin Kumar N**

* Computer Science Engineer
* AI | Full Stack | AWS | Blockchain Enthusiast

GitHub: [https://github.com/NithinGowda67](https://github.com/NithinGowda67)

---

## ⭐ Support

If you like this project, **star the repository** and share your feedback.
