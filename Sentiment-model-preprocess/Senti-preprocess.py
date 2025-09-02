import sys
import os

# 가상 환경의 site-packages 경로를 추가
venv_path = "/opt/anaconda3/envs/roatG/lib/python3.8/site-packages"
sys.path.append(venv_path)

from konlpy.tag import Mecab
from soynlp.normalizer import repeat_normalize

# Initialize tools
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
                    # print(f"Skipping line due to invalid score: {line.strip()}")
    return sentiword_dict

# 텍스트 전처리 함수
def preprocess_text(text):
    # 불필요한 반복 정규화 (ㅋㅋㅋ -> ㅋㅋ)
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

# 리뷰 데이터 예제
reviews = [
    "게임이 정말 재미있습니다. 캐릭터 디자인이 아주 마음에 들어요. 😊",
    "업데이트 이후에 버그가 너무 많아졌어요. 그래픽은 좋은데 플레이하기 힘들어요. 😑",
    "스토리가 너무 감동적이에요. 유저 인터페이스도 편리하고 좋아요. 💜",
    "이 게임은 정말 갓겜입니다! 혜자게임이에요.",
    "정말 지루하고 짜증나는 게임이네요. 망겜입니다."
]

# 각 리뷰에 대해 감성 점수 계산
review_scores = []
for review in reviews:
    preprocessed_review = preprocess_text(review)
    score, token_scores = calculate_sentiment_score(preprocessed_review, sentiword_dict)
    review_scores.append((review, preprocessed_review, score, token_scores))

# 결과 출력
for review, preprocessed_review, score, token_scores in review_scores:
    print(f"Original Review: {review}")
    print(f"Preprocessed Review: {preprocessed_review}")
    print(f"Sentiment Score: {score}")
    print("Token scores:")
    for token, token_score in token_scores:
        print(f"  {token}: {token_score}")
    print()
