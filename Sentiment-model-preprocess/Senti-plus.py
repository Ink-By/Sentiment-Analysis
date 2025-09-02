import sys
import os
import pandas as pd
from konlpy.tag import Mecab
from soynlp.normalizer import repeat_normalize

# 가상 환경의 site-packages 경로를 추가
venv_path = "/opt/anaconda3/envs/roatG/lib/python3.8/site-packages"
sys.path.append(venv_path)

# Mecab 초기화
mecab = Mecab(dicpath="./mecab-dict/mecab-ko-dic-2.1.1-20180720")

# SentiWord_Dict.txt 파일 읽기
def load_sentiword_dict(filepath):
    sentiword_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                try:
                    score = int(parts[1])
                    sentiword_dict[word] = score
                except ValueError:
                    pass
    return sentiword_dict

# 텍스트 전처리 함수
def preprocess_text(text):
    if not isinstance(text, str):
        text = ''
    normalized_sent = repeat_normalize(text, num_repeats=2)
    return normalized_sent

# 감성 점수 계산 함수
def calculate_sentiment_score(text, sentiword_dict):
    tokens = mecab.morphs(text)
    score = 0
    token_scores = []
    for token in tokens:
        if token in sentiword_dict:
            token_score = sentiword_dict[token]
            score += token_score
            token_scores.append((token, token_score))
        else:
            token_scores.append((token, 0))
    return score, token_scores

# SentiWord_Dict.txt 파일 경로
filepath = './datasets/SentiWord_Dict.txt'
sentiword_dict = load_sentiword_dict(filepath)

# 엑셀 파일들을 포함하는 폴더 경로
folder_path = './datasets/review_3m'
output_folder = './senti-result-감성사전-3m'  # 결과 저장 폴더 경로

# 결과 저장 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내 모든 엑셀 파일 처리
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_excel(file_path)

        # 각 리뷰에 대해 감성 점수 계산
        review_scores = []
        for idx, row in df.iterrows():
            review = row['content']
            preprocessed_review = preprocess_text(review)
            score, token_scores = calculate_sentiment_score(preprocessed_review, sentiword_dict)

            # 가중치 적용 (star와 thumbs)
            star_weight = row['score'] / 5.0  # star는 1에서 5 사이의 값
            thumbs_weight = row['thumbsUpCount'] / 100.0  # thumbs는 상대적인 가중치
            if row['thumbsUpCount'] == 0:
                thumbs_weight = 1

            weighted_score = score * star_weight * thumbs_weight
            senti = 0
            if weighted_score > 0:
                senti = 1
            elif weighted_score < 0:
                senti = -1

            star_senti = 0
            if row['score'] <= 2:
                star_senti = -1
            elif row['score'] >= 4:
                star_senti = 1

            review_scores.append((senti, star_senti))

        # 'senti'와 'star_senti' 열을 기존 데이터프레임에 추가
        df['senti'] = [s[0] for s in review_scores]
        df['star_senti'] = [s[1] for s in review_scores]

        # 새로운 파일명 생성 (senti_를 파일명 앞에 추가)
        new_filename = f"3month_senti_{filename}"
        output_path = os.path.join(output_folder, new_filename)

        # 결과를 새로운 엑셀 파일로 저장
        df.to_excel(output_path, index=False)
        print(f"Sentiment analysis completed for {filename}. Results saved to {output_path}")
