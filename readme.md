# Telegram Ecommerce NER

## Description
This project is designed for Named Entity Recognition (NER) in the context of Telegram data. It utilizes machine learning techniques to process and analyze data, enabling the identification of entities within text, such as products, prices, and locations.

## Project Overview
The **Telegram Ecommerce NER** project is designed to facilitate Named Entity Recognition (NER) in the context of Telegram data. This project leverages advanced machine learning techniques to process and analyze textual data, enabling the identification and classification of various entities such as products, prices, and locations within messages.

### Key Features:
- **Named Entity Recognition**: The project implements NER capabilities to accurately identify and label entities in Telegram messages, enhancing the understanding of user interactions and content.
- **Data Scraping**: Utilizing the Telethon library, the project can scrape data from Telegram, allowing for the collection of relevant messages for analysis.
- **Model Comparison**: The project includes functionalities for comparing different NER models, providing insights into their performance and effectiveness in recognizing entities.
- **Integration with Hugging Face Transformers**: The project is designed to work seamlessly with Hugging Face's Transformers library, enabling easy access to state-of-the-art models for training and evaluation.

### Installation and Usage:
Users can easily install the required dependencies and run the provided scripts to perform NER tasks or model comparisons. The project is structured to allow for straightforward integration and extension, making it suitable for both research and practical applications in the field of natural language processing.

## Features
- Named Entity Recognition (NER) capabilities for products, prices, and locations.
- Data scraping from Telegram using the Telethon library.
- Model comparison for evaluating the performance of different NER models.
- Easy integration with Hugging Face Transformers for model training and evaluation.

## Installation
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To use the project, you can run the relevant scripts located in the `scripts` directory. For example:
```bash
python scripts/ner_labelling.py
```
To compare models, you can run:
```bash
python scripts/model_comparison.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License.
