import streamlit as st
from transformers import pipeline, BertForQuestionAnswering, BertTokenizer
import torch

# Define each NLP use case function
def text_classification():
    st.title("Text Classification")
    text = st.text_area("Enter text to classify")
    if st.button("Classify"):
        classifier = pipeline('sentiment-analysis')
        result = classifier(text)
        st.write(result)

def named_entity_recognition():
    st.title("Named Entity Recognition")
    text = st.text_area("Enter text for NER")
    if st.button("Recognize"):
        ner = pipeline('ner')
        result = ner(text)
        st.write(result)

def question_answering():
    st.title("Question Answering")
    question = st.text_input("Enter your question")
    context = st.text_area("Enter context")
    if st.button("Get Answer"):
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        inputs = tokenizer(question, context, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start_index:answer_end_index+1]))
        st.write(answer)

def text_generation():
    st.title("Text Generation")
    prompt = st.text_area("Enter prompt for text generation")
    if st.button("Generate"):
        generator = pipeline('text-generation', model='gpt2')
        result = generator(prompt)
        st.write(result[0]['generated_text'])

def translation():
    st.title("Translation")
    text = st.text_area("Enter text to translate")
    if st.button("Translate"):
        translator = pipeline('translation_en_to_fr')
        result = translator(text)
        st.write(result[0]['translation_text'])

def summarization():
    st.title("Summarization")
    text = st.text_area("Enter text to summarize")
    if st.button("Summarize"):
        summarizer = pipeline('summarization')
        result = summarizer(text)
        st.write(result[0]['summary_text'])

def main():
    st.sidebar.title("NLP Applications")
    app_options = ["Text Classification", "Named Entity Recognition", "Question Answering", "Text Generation", "Translation", "Summarization"]
    app_choice = st.sidebar.selectbox("Choose an application", app_options)

    if app_choice == "Text Classification":
        text_classification()
    elif app_choice == "Named Entity Recognition":
        named_entity_recognition()
    elif app_choice == "Question Answering":
        question_answering()
    elif app_choice == "Text Generation":
        text_generation()
    elif app_choice == "Translation":
        translation()
    elif app_choice == "Summarization":
        summarization()

if __name__ == '__main__':
    main()
