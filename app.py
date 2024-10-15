import gradio as gr
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
import string

# Ensure necessary NLTK data is downloaded
nltk.download(['stopwords', 'punkt'])

# Predefined text inputs for users to choose from
predefined_texts = {
    "Sample Text 1": "This is a sample text to demonstrate the text analysis capabilities of this tool. Feel free to modify this text and see the results.",
    "Sample Text 2": "Another example text to showcase the functionality of the text analysis tool. Experiment with different texts to see how the analysis changes.",
    "Sample Text 3": "Text analysis tools can provide insights into sentiment, keywords, and summaries. Try using different texts to explore these features."
}

# Function to summarize text using the LSA algorithm from the Sumy library
def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# Function to preprocess text (lowercase, remove punctuation and stopwords)
def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    return " ".join(word for word in text.split() if word not in stop_words)

# Function to perform sentiment analysis using TextBlob
def sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity

# Function to generate a word cloud from the text
def generate_wordcloud(text):
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(stopwords=stop_words, background_color="white", width=800, height=400).generate(text)
    wordcloud_path = "wordcloud.png"
    wordcloud.to_file(wordcloud_path)
    return wordcloud_path

# Function to extract top keywords using TF-IDF
def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    indices = X[0].toarray().argsort()[0, -10:][::-1]
    features = vectorizer.get_feature_names_out()
    return ", ".join(features[i] for i in indices)

# Main function to handle the text analysis
def analyze_text(selected_text, user_text, sentence_count):
    try:
        text = user_text if user_text else predefined_texts[selected_text]
        processed_text = preprocess_text(text)
        sentiment = sentiment_analysis(processed_text)
        sentiment_result = (
            f"Positive Sentiment 🙂 (Score: {sentiment:.2f})" if sentiment > 0 else
            f"Negative Sentiment 😟 (Score: {sentiment:.2f})" if sentiment < 0 else
            f"Neutral Sentiment 😐 (Score: {sentiment:.2f})"
        )
        wordcloud_path = generate_wordcloud(processed_text)
        keywords_result = extract_keywords(processed_text)
        summary = summarize_text(text, sentence_count=sentence_count)
        return sentiment_result, wordcloud_path, keywords_result, summary
    except Exception as e:
        return f"An error occurred during analysis: {e}", None, None, None

# Gradio Interface to create the web app
iface = gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.Dropdown(choices=list(predefined_texts.keys()), label="Select Predefined Text 💬", value="Sample Text 1"),
        gr.Textbox(lines=5, placeholder="Or enter your own text here...", label="User Text Input ✍🏼"),
        gr.Slider(minimum=1, maximum=10, value=3, step=1, label="🔢 Number of Summary Sentences")
    ],
    outputs=[
        gr.Textbox(label="📊 Sentiment Analysis"),
        gr.Image(type="filepath", label="🌥️ Word Cloud"),
        gr.Textbox(label="🔑 Top Keywords"),
        gr.Textbox(label="📝 Text Summary")
    ],
    title="Linguistic Lens 🔍",
    description="A Text Analysis Tool 🛠️"
)

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch(share=True)
