# ESG Report Classification Tool

Author : Gopal Jamnal, 2025

This is an AI-powered tool for classifying **ESG (Environmental, Social, Governance)** reports using **BERT-based Natural Language Processing**. The tool allows you to input a URL to an ESG report, and it processes the document to provide insights into the classification of various ESG-related topics.

## Features

- Classify ESG reports using a pre-trained BERT model for text classification.
- Easy-to-use web interface built with React.
- Real-time results visualization (bar chart) for the classification scores.
- Integration with a Flask backend to process ESG reports.
- Displays progress during processing to enhance user experience.

## Example of Using the Tool
1. Enter the URL of an ESG report in the provided input box. For example, you can try:

    McDonald's CSR Report
    Amazon CSR Report
    Newmont Mining Report

2. Click the "Classify Report" button, and the system will process the document and display the classification results on a bar chart.

## Tech Stack

- **Frontend**: React, Plotly.js (for charting), Axios (for API requests).
- **Backend**: Python, Flask, Hugging Face's `transformers` library (for BERT model), Tika (for PDF parsing).
- **API**: Flask API for interacting with the frontend.

## Setup and Installation

### Prerequisites

- Node.js (for React frontend)
- Python 3.x (for Flask backend)
- `pip` (for Python package management)

### Frontend Setup (React)

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>/frontend


