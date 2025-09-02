import pickle
from tensorflow.keras.models import load_model
import os
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Mecab
import pandas as pd

# Mecab 초기화
mecab = Mecab(dicpath="./mecab-dict/mecab-ko-dic-2.1.1-20180720")

# Tokenizer 로드
with open('./keras-model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 저장된 모델 로드
loaded_model = load_model('./keras-model/best_model.keras')

# 불용어 정의
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']

# 최대 문장 길이
max_len = 60

# 엑셀 파일 읽기
excel_path = './datasets/paper-datasets/review_레이드 그림자의 전설.xlsx'
df = pd.read_excel(excel_path)

#텍스트 전처리
def preprocess_text(text):
    # 불필요한 반복 정규화 (ㅋㅋㅋ -> ㅋㅋ)
    if not isinstance(text, str):
        text = ''
    normalized_sent = repeat_normalize(text, num_repeats=2)
    return normalized_sent
# 감정 예측 함수
def sentiment_predict(new_sentence):
    # 특수 문자 제거
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    # 토큰화
    new_sentence = mecab.morphs(new_sentence)
    # 불용어 제거
    new_sentence = [word for word in new_sentence if not word in stopwords]
    # 정수 인코딩
    encoded = tokenizer.texts_to_sequences([new_sentence])
    # 패딩 처리
    pad_new = pad_sequences(encoded, maxlen=max_len)
    # 예측
    score = float(loaded_model.predict(pad_new))
    
    # 결과 반환
    if score > 0.5:
        sentiment = 1
        probability = score * 100
    else:
        sentiment = -1
        probability = (1 - score) * 100
    
    return sentiment, probability

# 엑셀 파일 읽기 (예: datasets/paper-datasets/review_쿠키런 킹덤.xlsx)
file_path = './datasets/paper-datasets/review_쿠키런 킹덤.xlsx'
df = pd.read_excel(file_path)

# 각 리뷰에 대해 감정 분석 수행
sentiments = []
probabilities = []

for idx, row in df.iterrows():
    content = row['content']  # 'content' 열의 내용
    sentiment, probability = sentiment_predict(content)
    
    # 결과를 리스트에 저장
    sentiments.append(sentiment)
    probabilities.append(probability)

# 기존 데이터프레임에 새로운 열 추가
df['sentiment'] = sentiments
df['probability'] = probabilities

# 결과를 새로운 엑셀 파일로 저장
output_folder = './senti-result'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file_path = os.path.join(output_folder, 'sentiment_analysis_with_original_data.xlsx')
df.to_excel(output_file_path, index=False)

print(f"Sentiment analysis completed. Results saved to {output_file_path}")
