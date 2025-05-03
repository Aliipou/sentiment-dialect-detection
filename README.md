# Multilingual Sentiment Analysis and Persian Dialect Detection

This project is an intelligent natural language processing system that performs sentiment analysis on multilingual texts and dialect detection on Persian texts. It uses state-of-the-art transformer models and provides both a web interface and API for easy integration.

## Features

- **Sentiment Analysis**:
  - Detects positive, negative, or neutral sentiment in texts
  - Supports multiple languages including Persian, English, French, German, Spanish, and more
  - Provides confidence scores for predictions

- **Persian Dialect Detection**:
  - Identifies common Persian dialects (Tehrani, Isfahani, Shirazi, Mashhadi, and others)
  - Uses both neural network and rule-based approaches
  - Provides confidence scores for predictions

- **User Interface**:
  - Web-based UI built with Streamlit
  - Support for single text analysis and batch processing
  - File upload capability (TXT and CSV files)
  - Visualization of results with charts and graphs
  - Download analysis results in CSV and JSON formats

- **API**:
  - RESTful API built with FastAPI
  - Swagger documentation
  - Endpoints for sentiment analysis, dialect detection, and combined analysis
  - Batch processing capabilities

## Installation

### Prerequisites

- Python 3.7 or higher
- Sufficient disk space for model downloads (~1GB)
- 4GB RAM minimum (8GB+ recommended)
- GPU support optional but recommended for faster training and inference

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-dialect-detection.git
cd sentiment-dialect-detection

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# OR
venv\Scripts\activate     # For Windows

# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

The easiest way to run the project is using the main runner script, which handles data preparation, model training, and launching both the API and web interface:

```bash
python main_runner.py
```

After running this command:
- The API will be accessible at http://localhost:8000
- The web interface will be accessible at http://localhost:8501

### API Usage

You can interact with the API using curl commands or any HTTP client:

```bash
# Sentiment analysis
curl -X POST "http://localhost:8000/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was excellent, I really enjoyed it.", "language": "en"}'

# Dialect detection (Persian only)
curl -X POST "http://localhost:8000/dialect" \
  -H "Content-Type: application/json" \
  -d '{"text": "سلام داداش، چطوری؟ دیشب کجا بودی؟"}'

# Combined analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "این فیلم عالی بود", "language": "fa"}'

# Batch processing
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["این فیلم عالی بود", "The movie was great"], "language": "fa"}'
```

For detailed API documentation, visit http://localhost:8000/docs after starting the server.

### Web Interface Usage

The web interface provides an intuitive way to interact with the system:

1. Open your browser and navigate to http://localhost:8501
2. Select the analysis mode (single text, batch, or file upload)
3. Choose the language of your text
4. Enter or upload your text
5. Click the analyze button to see the results
6. Download results in CSV or JSON format if needed

## Advanced Usage

### Custom Training

You can train the models with your own data by:

1. Preparing your data in CSV format with 'text' and 'sentiment'/'dialect' columns
2. Placing your data files in the 'data' directory
3. Running the training process:
   ```bash
   python main_runner.py --skip-setup
   ```

### Command Line Options

The `main_runner.py` script provides several options:

```bash
# Skip library installation
python main_runner.py --skip-setup

# Skip model training (use pre-trained models if available)
python main_runner.py --skip-training

# Only launch the API
python main_runner.py --api-only

# Only launch the web interface
python main_runner.py --ui-only
```

### Individual Component Execution

You can run each component separately:

```bash
# Data preprocessing
python data_preprocessing.py

# API server
uvicorn sentiment_api:app --host 0.0.0.0 --port 8000

# Web interface
streamlit run streamlit_app.py
```

## Technical Details

### Models

- **Persian Sentiment Analysis**: Uses ParsBERT (BERT model pre-trained on Persian texts)
- **Multilingual Sentiment Analysis**: Uses XLM-RoBERTa (capable of handling 100+ languages)
- **Persian Dialect Detection**: Uses a fine-tuned BERT model with Persian pre-training

### Data Processing

- **Persian Texts**: Uses Hazm library for normalization and tokenization
- **Multilingual Texts**: Uses language-specific preprocessing techniques
- **Dialect Features**: Includes word usage patterns, specific dialect markers, and grammatical structures

### Performance Considerations

- First-time model downloads may take time depending on your internet connection
- Training requires significant computational resources, especially for the multilingual model
- Inference is relatively fast, especially if a GPU is available
- The API can handle concurrent requests with reasonable performance

## Project Structure

```
sentiment-dialect-detection/
├── data/                                # Training data directory
│   ├── persian_sentiment_data.csv      # Persian sentiment data
│   ├── english_sentiment_data.csv      # English sentiment data
│   ├── persian_dialect_data.csv        # Persian dialect data
│   └── french_sentiment_data.csv       # French sentiment data
│
├── models/                             # Trained models directory
│   ├── sentiment_model/                # Persian sentiment analysis model
│   ├── dialect_model/                  # Persian dialect detection model
│   └── multilingual_model/             # Multilingual sentiment analysis model
│
├── sentiment_model.py                  # Persian sentiment analysis model implementation
├── dialect_detector.py                 # Persian dialect detection model implementation
├── multilingual_model.py               # Multilingual sentiment analysis model implementation
├── data_preprocessing.py               # Data preprocessing
├── sentiment_api.py                    # API for using the models
├── streamlit_app.py                    # Web user interface
├── main_runner.py                      # Main script for launching the project
├── api_config.json                     # API settings
├── README.md                           # Project documentation
└── requirements.txt                    # Required libraries
```

## Expanding the Project

There are several ways you can expand this project:

- Adding support for more languages
- Implementing more detailed emotion analysis beyond positive/negative/neutral
- Integrating with social media APIs for real-time analysis
- Creating a more comprehensive dialect detection system with more regional variations
- Implementing a deployment pipeline for production environments

## Troubleshooting

- **Model download issues**: Check your internet connection and firewall settings
- **GPU not being used**: Ensure PyTorch is installed with CUDA support
- **API connection errors**: Verify ports are not in use by other applications
- **Out of memory errors**: Reduce batch size or use CPU-only mode if memory is limited

## Contributors

- Author: [Your Name]
- Email: [Your Email]

## License

This project is released under the MIT License.