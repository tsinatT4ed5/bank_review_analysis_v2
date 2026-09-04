# 🏦 Ethiopian Banks Customer Review Scraping & Data Preprocessing

A data collection and preprocessing project focused on gathering and preparing **Google Play Store customer reviews** for three Ethiopian banks: **Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank**.

The project demonstrates an end-to-end workflow for collecting real-world customer feedback, cleaning raw review data, and preparing a structured dataset for subsequent sentiment and exploratory analysis.

---

## 📌 Project Overview

Customer reviews provide valuable insights into users' experiences with banking applications. However, raw review data often contains duplicates, inconsistent date formats, and other quality issues that need to be addressed before analysis.

For this task, customer reviews were collected from the **Google Play Store** using the `google-play-scraper` Python library and then processed using **Pandas**.

The workflow consisted of:

**Data Collection → Data Cleaning → Duplicate Removal → Date Normalization → Quality Checking → Clean Dataset**

---

## 🎯 Objectives

The main objectives of this task were to:

* Collect customer reviews for selected Ethiopian banking applications
* Build a reproducible web-scraping workflow
* Combine reviews from multiple banking applications
* Identify and remove duplicate reviews
* Normalize date fields for consistent analysis
* Produce a clean dataset ready for further analysis

---

## 🏦 Banks Covered

The project collected customer reviews for:

* **Commercial Bank of Ethiopia (CBE)**
* **Bank of Abyssinia (BOA)**
* **Dashen Bank**

---

## 🛠️ Tools & Technologies

| Tool / Technology       | Purpose                                |
| ----------------------- | -------------------------------------- |
| **Python**              | Data collection and preprocessing      |
| **google-play-scraper** | Google Play Store review extraction    |
| **Pandas**              | Data cleaning and transformation       |
| **Git & GitHub**        | Version control and project management |

---

## 📊 Data Collection Results

The scraping process collected:

* **1,362 raw customer reviews**
* **1,187 unique reviews after cleaning**
* Reviews from **3 Ethiopian banks**

The reduction in the final dataset was primarily due to the removal of duplicate records, resulting in a cleaner and more reliable dataset for downstream analysis.

### Dataset Summary

| Stage                             | Number of Reviews |
| --------------------------------- | ----------------: |
| Raw reviews collected             |         **1,362** |
| Duplicate/invalid records removed |           **175** |
| Final clean reviews               |         **1,187** |

The final dataset is available at:

```text
data/clean_bank_reviews.csv
```

---

## 🧹 Data Preprocessing

The preprocessing workflow included:

### 1. Duplicate Removal

Duplicate reviews were identified and removed to avoid counting the same customer feedback multiple times.

### 2. Date Normalization

Review dates were standardized into a consistent format to make the dataset suitable for chronological and trend analysis.

### 3. Data Quality Checking

The resulting dataset was checked to ensure that the cleaned records were structured consistently and ready for subsequent analytical tasks.

---

## 📁 Repository Structure

```text
bank-review-scraping/
│
├── 📂 data/
│   └── clean_bank_reviews.csv
│
├── 📜 scrape_reviews.py
├── 📜 preprocess_data.py
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📖 README.md
```

---

## 🔄 Reproducible Workflow

The project is organized so that the data collection and preprocessing stages can be reproduced using the provided scripts.

### Scrape Reviews

```bash
python scrape_reviews.py
```

### Preprocess Data

```bash
python preprocess_data.py
```

The required Python dependencies are listed in:

```text
requirements.txt
```

---

## 🌿 Version Control

The project was managed using **Git and GitHub**, including:

* A dedicated `task-1` branch
* Descriptive commits
* Organized scripts and data directories
* A `.gitignore` file for repository hygiene

This structure makes the project easier to maintain, reproduce, and extend.

---

## 🚀 Next Phase

The cleaned dataset provides a foundation for further customer feedback analysis, including:

* 😊 **Sentiment analysis**
* ⭐ **Rating distribution analysis**
* 🔍 **Common complaint identification**
* 💬 **Customer feedback categorization**
* 📊 **Bank-by-bank comparison**
* 📈 **Trend analysis over time**
* 🤖 **Natural Language Processing (NLP)**

---

## 👩‍💻 Author

**Tsinat Demelash**

Data Science Graduate | Data Analytics | Business Intelligence

🔗 **Portfolio:** https://tsinatt4ed5.github.io/my_portfolio/

🔗 **GitHub:** https://github.com/tsinatT4ed5

---

⭐ This project demonstrates practical experience in **data collection, web scraping, data cleaning, preprocessing, and reproducible data workflows**.
