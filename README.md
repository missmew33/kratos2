# Scientometric Production Analyzer

**A Streamlit-based tool for calculating bibliometric diversity and parity indices.**

## Abstract

This repository contains the source code for the **Scientometric Production Analyzer**, a web-based application designed to assist researchers in evaluating the academic rigor of bibliographic datasets. The tool processes author metadata to calculate two primary indices: the **Collaboration Diversity Index (CDI)** and the **Gender Parity Index (GPI)**.

The application separates calculation logic from visualization, allowing for modular use in both web environments (Streamlit) and computational notebooks (Google Colab/Jupyter).

---

## Methodological Framework

The algorithms implemented in this tool are grounded in established scientometric and ecological statistical methods.

### 1. Collaboration Diversity Index (CDI)
To measure the heterogeneity of international collaborations, the tool employs **Simpson's Diversity Index** ($1 - D$).

* **Formula:** $CDI = 1 - \sum (p_i^2)$
* **Interpretation:** Values range from 0 to 1. A value closer to **1.0** indicates high heterogeneity (authors are distributed across many different countries), while a value closer to **0** indicates homogeneity (concentration in a single country).
* **Reference:** Simpson, E. H. (1949). Measurement of diversity. *Nature*, 163, 688.

### 2. Gender Parity Index (GPI)
To assess gender balance within the dataset, the tool calculates the convergence to absolute parity based on probabilistic name inference.

* **Formula:** $GPI = 1 - |P_m - P_f|$
    * Where $P_m$ is the proportion of male authors and $P_f$ is the proportion of female authors.
* **Interpretation:** A value of **1.0** represents perfect statistical parity (50/50 split). Lower values indicate a greater disparity between genders.
* **Note:** Gender inference is performed using the `gender-guesser` library. Results are probabilistic and limited to binary classifications (Male/Female) for the purpose of this specific metric.

---

## Repository Structure

```text
/
├── app.py               # Main application entry point (Frontend + Logic)
├── requirements.txt     # Python dependencies for reproduction
└── README.md            # Documentation

Installation and Reproducibility
Prerequisites
Python 3.8+

pip (Python Package Installer)

Local Deployment
Clone the repository:

Bash

git clone [https://github.com/your-username/scientometric-analyzer.git](https://github.com/your-username/scientometric-analyzer.git)
cd scientometric-analyzer
Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Bash

streamlit run app.py
Cloud Deployment
This repository is optimized for Streamlit Community Cloud.

Push this code to a GitHub repository.

Connect your repository to Streamlit Cloud.

Select app.py as the main file.

Data Specifications
The application accepts .csv or .txt (TSV) files. The dataset must contain, at a minimum, columns representing:

Author Names: Full names are preferred for accurate gender inference.

Affiliations/Countries: Country names (e.g., "USA", "France", "Chile"). The tool utilizes country_converter to standardize these to ISO3 codes automatically.

Technologies Used
Streamlit: Web application framework.

Pandas: Data manipulation and analysis.

Plotly: Interactive scientific visualization.

Gender-Guesser: Name-based gender inference.

Country-Converter: Geopolitical entity standardization.

License MIT

Author: María Dolores  Gonzalez Barbado ORCID 0000-0002-4213-2090
