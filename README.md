# 🛒 AI-Powered E-Commerce Intelligence System

An AI-powered conversational Business Intelligence platform that allows users to analyze e-commerce data using **natural language queries** instead of writing SQL manually.

The system uses **LangChain** and **Groq API (LLM)** to convert natural-language questions into SQL queries, execute them on a **MySQL database**, generate business insights, and display results through interactive visualizations.

---

## 📌 Overview

Traditional Business Intelligence systems often require users to have knowledge of SQL and data analytics tools. This makes it difficult for non-technical users to extract insights from large amounts of e-commerce data.

This project provides a conversational interface where users can simply ask questions such as:

```text
Show total revenue for each category
```

or

```text
Which city has the most orders?
```

The system processes the question, generates the required SQL query, retrieves the data from MySQL, generates AI-powered insights, and presents the results using charts and visualizations.

---

## 🎯 Objectives

* Enable natural-language querying of e-commerce data.
* Reduce dependency on SQL knowledge.
* Automatically generate SQL queries using an LLM.
* Execute generated queries on a MySQL database.
* Generate meaningful business insights.
* Provide interactive dashboards and visualizations.
* Generate charts dynamically based on query results.
* Provide downloadable PDF and Excel reports.
* Maintain chat sessions and conversation history.
* Evaluate AI-generated responses using multiple performance metrics.

The main goal is to make Business Intelligence more accessible to non-technical users.

---

## ✨ Features

### 💬 Conversational AI

Ask business questions using natural language without writing SQL manually.

### 🧠 Natural Language to SQL

The LangChain-based AI agent uses the Groq API to convert natural-language questions into SQL queries.

Example:

```text
User:
Show total revenue for each category
```

Generated SQL:

```sql
SELECT category,
       SUM(price * quantity) AS revenue
FROM products p
JOIN orders o
ON p.product_id = o.product_id
GROUP BY category;
```

### 📊 Interactive Dashboard

The Streamlit dashboard provides an overview of e-commerce business performance.

### 📈 Dynamic Visualizations

The system dynamically generates:

* Line charts
* Bar charts
* Pie charts
* Analytical visualizations

### 💡 AI-Generated Business Insights

The system generates insights such as:

* Top-performing products and customers
* Revenue trends
* Region-wise sales performance
* Category-wise sales performance
* Order status and fulfilment analysis

### 💾 Chat Session Management

The system stores chat sessions, messages, query results, chart data, and timestamps.

### 📄 Report Generation

The application supports downloadable:

* PDF reports
* Excel reports

### 🧪 Model Evaluation

The system evaluates AI responses using:

* Accuracy
* Precision
* Recall
* F1 Score
* Semantic Similarity
* Task Success Rate
* Hallucination Rate

---

## 🏗️ System Workflow

```text
User
  │
  ▼
Natural Language Query
  │
  ▼
Streamlit Interface
  │
  ▼
LangChain AI Agent
  │
  ▼
Groq API / LLM
  │
  ▼
SQL Query Generation
  │
  ▼
SQLAlchemy
  │
  ▼
MySQL Database
  │
  ▼
Pandas Data Processing
  │
  ├───────────────┐
  ▼               ▼
Visualization   AI Insights
  │               │
  └───────┬───────┘
          ▼
      Final Result
```

The documented workflow follows Streamlit → LangChain → Groq → MySQL through SQLAlchemy → Pandas → visualization and AI-generated insights.

---

## 🧰 Technology Stack

| Category             | Technology                 |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Frontend             | Streamlit                  |
| Database             | MySQL                      |
| ORM                  | SQLAlchemy                 |
| Data Processing      | Pandas                     |
| AI Framework         | LangChain                  |
| LLM                  | Groq API                   |
| Visualization        | Plotly                     |
| Visualization        | Matplotlib                 |
| Data Source          | Custom & Synthetic Dataset |

---

## 🗄️ Database Structure

The project uses a MySQL database named:

```text
ecommerce_ai
```

### Users Table

Stores customer information.

```text
user_id
name
email
city
```

### Products Table

Stores product information.

```text
product_id
title
price
category
```

### Orders Table

Stores transaction information.

```text
order_id
user_id
product_id
quantity
order_date
region
payment_status
```

### Chat Sessions Table

Stores chat sessions.

```text
session_id
title
created_at
updated_at
```

### Chat History Table

Stores conversations and analytical results.

```text
id
session_id
role
message
data
chart_type
chart_data
created_at
```

---

## 📦 Dataset

The project uses a **custom-designed and synthetic e-commerce dataset** representing realistic e-commerce scenarios.

The dataset contains information related to:

* Users
* Products
* Orders
* Transactions
* Customer locations
* Product categories
* Payment status

The dataset is stored in the MySQL `ecommerce_ai` database.

---

## 🔄 Application Flow

### 1. Enter Query

The user enters a natural-language question through the Streamlit interface.

```text
Which city has the most orders?
```

### 2. Process Query

The LangChain-based AI agent processes the user's question.

### 3. Generate SQL

The Groq-powered LLM converts the question into SQL.

### 4. Execute Query

SQLAlchemy executes the generated SQL query on MySQL.

### 5. Process Data

The retrieved data is processed using Pandas.

### 6. Generate Insights

The LLM generates business insights from the query results.

### 7. Generate Visualization

The system creates an appropriate visualization using Plotly or Matplotlib.

### 8. Evaluate Response

The response can be evaluated using the model evaluation module.

---

## 📊 Example Queries

```text
Show total revenue for each category
```

```text
Which city has the most orders?
```

```text
Show revenue trends over time
```

```text
Which products are performing the best?
```

```text
Show region-wise sales performance
```

```text
Show category-wise sales
```

---

## 🧪 Model Evaluation

The system includes a predefined test suite of e-commerce analytical queries.

### Evaluation Metrics

| Metric              | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| Accuracy            | Measures the proportion of correct responses               |
| Precision           | Measures relevant predicted responses                      |
| Recall              | Measures retrieval of relevant results                     |
| F1 Score            | Harmonic mean of precision and recall                      |
| Semantic Similarity | Measures similarity between generated and expected answers |
| Task Success Rate   | Measures successful completion of analytical tasks         |
| Hallucination Rate  | Identifies incorrect or unsupported information            |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <PROJECT_FOLDER>
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create MySQL Database

```sql
CREATE DATABASE ecommerce_ai;
```

Configure your MySQL connection according to your local credentials.

### 5. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key

DB_HOST=localhost
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=ecommerce_ai
```

> **Important:** Never upload your API key or database password to GitHub.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

---

## 💻 System Requirements

### Hardware

| Component | Requirement             |
| --------- | ----------------------- |
| Processor | Intel Core i3 or higher |
| RAM       | 8 GB                    |
| Storage   | 256 GB SSD              |
| Internet  | Required for API access |
| Display   | 1366 × 768 or higher    |

### Software

| Software         | Requirement                    |
| ---------------- | ------------------------------ |
| Operating System | Windows 10 / Windows 11        |
| Python           | Python 3.x                     |
| Database         | MySQL                          |
| Frontend         | Streamlit                      |
| IDE              | Visual Studio Code             |
| Browser          | Google Chrome / Microsoft Edge |

---

## 📁 Project Structure

```text
AI-Powered-E-Commerce-Intelligence/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── data/
│   └── synthetic_data/
│
├── database/
│   └── database_setup.sql
│
├── modules/
│   ├── database.py
│   ├── ai_agent.py
│   ├── query_generator.py
│   ├── data_processing.py
│   ├── visualization.py
│   └── report_generator.py
│
└── evaluation/
    └── evaluation.py
```

---

## 🚀 Future Enhancements

* FastAPI / REST API integration
* Real-time data streaming
* Improved handling of complex and ambiguous queries
* Integration with larger real-world datasets
* User authentication
* Role-based access control
* Cloud deployment
* Advanced analytics and visualization

---

## 👩‍💻 Author

**Tanzeem Hundekari**

**Project:** AI-Powered E-Commerce Intelligence System

---

## ⭐ Project Highlights

* 🤖 Natural Language → SQL
* 🧠 LLM-powered Business Intelligence
* 💬 Conversational Data Analysis
* 🗄️ MySQL Database
* 🔗 LangChain AI Agent
* ⚡ Groq API
* 📊 Streamlit Dashboard
* 📈 Plotly & Matplotlib
* 🐼 Pandas Data Processing
* 📄 PDF & Excel Reports
* 💾 Chat Session Management
* 🧪 AI Model Evaluation
* 💡 Automated Business Insights

---

## 📜 License

This project was developed for academic purpose.
