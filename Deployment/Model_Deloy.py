# streamlit run Model_Deloy.py

# region LIBRARY
import io
import os
import re
import regex
import torch
import streamlit as st
import torch.nn as nn
import pandas as pd
from pytube import YouTube
import speech_recognition as sr
from PIL import Image
from gensim.utils import simple_preprocess
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
# endregion LIBRARY

# region Model PhoBert
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x

@st.cache_resource
def load_model(model_path):
    device = torch.device('cpu')
    model = SentimentClassifier(n_classes=11)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, device

@st.cache_data
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return tokenizer

@st.cache_data
def load_classname():
    class_names =['Chính trị Xã hội', 'Dự báo thời tiết', 'Kinh tế',
                     'Môi trường', 'Nông nghiệp', 'Pháp luật', 'Sức khỏe',
                     'Thế giới', 'Thể thao', 'Văn hóa', 'Giáo dục']
    return class_names

def infer(text, tokenizer, models, class_names, max_len=256):
    for model in models:
        model.to(device)
        model.eval()

    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        all_outputs = []
        for model in models:
            output = model(input_ids, attention_mask)
            all_outputs.append(output)

        all_outputs = torch.stack(all_outputs)
        mean_output = torch.mean(all_outputs, dim=0)
        _, y_pred = torch.max(mean_output, dim=1)

    probabilities = torch.softmax(mean_output, dim=1)
    probabilities = probabilities.squeeze().cpu().numpy()

    df = pd.DataFrame({'Topic': class_names, 'Probability': probabilities})
    df = df.sort_values(by='Probability', ascending=False)
    df = df.reset_index(drop=True)
    
    predicted_class = df.iloc[0]['Topic']

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Topic", divider='rainbow')
        st.title(predicted_class)
        # st.write(text)

    with col2:
        st.subheader("Prob. distribution", divider='rainbow')
        st.dataframe(df.style.highlight_max(axis=0, subset=['Probability']))
    st.balloons()

# endregion PhoBert

# region Tien Xu Ly
@st.cache_data
def load_stopwords():
    stopwords = set()
    with open('stopwords.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    stopwords = [line.replace('\n', '') for line in lines]
    return stopwords

def remove_html(txt):
    return regex.sub(r'<[^>]*>', '', txt)

def remove_text(document):
    document = remove_html(document)
    document = ViTokenizer.tokenize(document)
    document = document.lower()
    document = regex.sub(
        r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', document)
    document = regex.sub(r'\s+', ' ', document).strip()
    document = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), document, flags=re.IGNORECASE)
    document = document.replace(u'"', u' ')
    document = document.replace(u'️', u'')
    document = document.replace('🏻', '')
    return document

def remove_stopwords(line, stopwords):
    words = []
    for word in line.strip().split():
        if word not in stopwords:
            words.append(word)
    return ' '.join(words)

def preprocess_text(lines, stopwords):
    lines = remove_text(lines)
    lines = simple_preprocess(lines)
    lines = ' '.join(lines)
    lines = remove_stopwords(lines, stopwords)
    return lines

stopwords = load_stopwords()
# endregion Tien Xu Ly

# region Dowload - Convert Video
def convert_video_to_text(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio

    temp_audio_path = "temp_audio.wav"
    audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
    recognizer = sr.Recognizer()

    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)

    audio_text = recognizer.recognize_google(audio_data, language='vi-VN')
    video.close()
    return audio_text

# def download_youtube_video(youtube_url):
#     yt = YouTube(youtube_url)
#     stream = yt.streams.filter(file_extension="mp4").first()
#     video_path = os.path.join("temp_youtube_video.mp4")
#     stream.download(filename="temp_youtube_video")
#     os.rename(stream.default_filename, video_path)
#     return video_path

def download_youtube_video(youtube_url, quality='360p'):
    yt = YouTube(youtube_url, 
    use_oauth=True,
    allow_oauth_cache=True)
    stream = yt.streams.filter(res=quality, file_extension="mp4").first()
    video_path = os.path.join("temp_youtube_video.mp4")
    stream.download(filename="temp_youtube_video")
    os.rename(stream.default_filename, video_path)
    return video_path
# endregion Dowload Video

# region Main Window

image = Image.open('banner.png')
show = st.image(image)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("https://coolbackgrounds.io/images/backgrounds/index/compute-ea4c57a4.png");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.subheader("", divider='rainbow')
st.subheader("📺System Television News Topic Classifier ", divider='rainbow')

models =[]
path = 'Model'
with st.spinner('Load model....'):
    model_filename = 'phobert_fold1.pth'
    model_path = os.path.join(path, model_filename)
    loaded_model, device = load_model(model_path)
    models.append(loaded_model)
    tokenizer = load_tokenizer()
    class_names = load_classname()
st.toast('Load model successfully', icon='🎉')

# st.sidebar.subheader("📺 MÔ HÌNH PHÂN LOẠI BẢN TIN THỜI SỰ TRUYỀN HÌNH")
# st.sidebar.text("Từ Thái Bảo - 1900222 :) ")
# st.sidebar.subheader("Hiệu suất các mô hình: ")
# st.sidebar.image("SoSanh.png", use_column_width=True)

if "visibility" not in st.session_state:
    st.session_state.horizontal = True

option = st.radio("Select option",
    ["Upload ⬆️", "URL 🔗", "Text 📃", "Micro 🎙️"],
    key="visibility",
    horizontal=st.session_state.horizontal)
st.write('\n')

if option == "Upload ⬆️":
    video_file = st.file_uploader("Choose Video", type=["mp4"])
    if video_file is not None:
        @st.cache_data
        def process_video(video):
            st.write("Title video:", video_file.name)
            video_path = os.path.join("temp_video.mp4")

            with open(video_path, "wb") as temp_video:
                temp_video.write(video_file.read())
            st.video(video_path)
            with st.spinner('Convert to Text'):
                text = convert_video_to_text(video_path)
            return text
        
        text = process_video(video_file)

        if st.button("Predict Topic"):
            text1 = preprocess_text(text, stopwords)
            with st.spinner('Model Classification'):
                infer(text1, tokenizer, models, class_names)
        
            os.remove("temp_audio.wav")
            os.remove("temp_video.mp4")

if option == "URL 🔗":
    st.write('\n')
    youtube_url = st.text_input("Enter the video URL from YouTube:")
    if youtube_url:
        try:
            @st.cache_data
            def url (youtube_url):
                yt = YouTube(youtube_url)
                st.write("Title video:", yt.title)
                st.write("Video author:", yt.author)
                stream = yt.streams.get_highest_resolution()

                buffer = io.BytesIO()
                stream.stream_to_buffer(buffer)
                video_bytes = buffer.getvalue()
                video_path = "temp_youtube_video.mp4"
                with open(video_path, "wb") as temp_video:
                    temp_video.write(video_bytes)
                st.video(video_path)

                with st.spinner('Convert to Text'):
                    text = convert_video_to_text(video_path)
                return text
            
            text = url (youtube_url)

            if st.button("Predict Topic "):
                with st.spinner('Convert to Text'):
                    # text = convert_video_to_text(video_path)
                    text1 = preprocess_text(text, stopwords)
                    with st.spinner('Model Classification'):
                        infer(text1, tokenizer, models, class_names)

                os.remove("temp_youtube_video.mp4")
                os.remove("temp_audio.wav")
        except Exception as e:
            st.error(f"Error!: {str(e)}")

if option == "Text 📃":
    with st.form(key='emotion_clf_form'):
        text = st.text_area("Enter documents here:")
        submit = st.form_submit_button(label='Predict')
        if submit:
            text1 = preprocess_text(text, stopwords)
            with st.spinner('Model Classification'):
                infer(text1, tokenizer, models, class_names)

if option == "Micro 🎙️":

    def recognize_speech():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            with st.spinner("Please say something.."):
                audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            return text.lower()
        except sr.UnknownValueError:
            st.write("Unable to recognize voice!")
            return None
        except sr.RequestError as e:
            st.write("Error connecting to Google Speech Recognition service {0}".format(e))
            return None
    
    if st.button("Micro 🎙️"):
        text = recognize_speech()
        if text is not None:
            st.subheader("You said:", divider='rainbow')
            st.info(text)
            st.subheader("", divider='rainbow')
            text1 = preprocess_text(text, stopwords)
            with st.spinner('Model Classification'):
                infer(text1, tokenizer, models, class_names)

# endregion Main Window