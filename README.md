# Linguistic Lens 🔍

Linguistic Lens is a comprehensive text analysis tool that provides insights into sentiment, keywords, and summaries of given texts. This tool leverages various NLP libraries to offer a user-friendly interface for text analysis.

## Features

- **Sentiment Analysis**: Determine the sentiment of the text (positive, negative, or neutral) using TextBlob.
- **Word Cloud Generation**: Visualize the most frequent words in the text with a word cloud.
- **Keyword Extraction**: Extract top keywords from the text using TF-IDF.
- **Text Summarization**: Summarize the text using the LSA algorithm from the Sumy library.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To launch the Gradio interface, run:

```bash
python app.py
```

This will start a web application where you can input text and receive analysis results.

## Predefined Texts

The application comes with predefined texts for quick analysis:

- **Sample Text 1**: Demonstrates the text analysis capabilities.
- **Sample Text 2**: Showcases the functionality of the tool.
- **Sample Text 3**: Provides insights into sentiment, keywords, and summaries.

## Input Options

- **Select Predefined Text**: Choose from predefined texts.
- **User Text Input**: Enter your own text for analysis.
- **Number of Summary Sentences**: Adjust the number of sentences in the summary.

## Outputs

- **Sentiment Analysis**: Displays the sentiment score and corresponding emoji.
- **Word Cloud**: Shows a word cloud image of the text.
- **Top Keywords**: Lists the top keywords extracted from the text.
- **Text Summary**: Provides a summary of the text.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Gradio](https://gradio.app/)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)
- [WordCloud](https://github.com/amueller/word_cloud)
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Sumy](https://github.com/miso-belica/sumy)
