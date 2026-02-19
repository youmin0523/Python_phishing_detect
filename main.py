"""
한국어 보이스 피싱 탐지 학습 스크립트 (TF-IDF + LinearSVC)


==== 전체 진행 절차 ====
1. CSV 데이터 로드 및 전처리
   - voice_phishing_text_label.csv 파일을 읽어옴
   - 'text'(텍스트)와 'label'(0=정상, 1=보이스피싱) 컬럼 확인
   - 결측치 제거 및 데이터 타입 변환


2. 데이터 분할
   - 전체 데이터를 학습용(80%)과 테스트용(20%)으로 분할
   - 보이스피싱과 정상 텍스트의 비율을 유지하여 분할(stratify)


3. 텍스트 벡터화 및 모델 학습
   - TF-IDF: 텍스트를 숫자 벡터로 변환 (단어의 중요도 계산)
   - LinearSVC: 선형 서포트 벡터 머신으로 분류 모델 학습
   - Pipeline으로 벡터화와 분류를 하나의 과정으로 연결


4. 모델 평가
   - 테스트 데이터로 모델 성능 평가
   - 정확도, 정밀도, 재현율, F1-score 출력
   - 혼동행렬로 분류 결과 시각화


5. 모델 저장
   - 학습된 모델을 .joblib 파일로 저장
   - 나중에 test.py에서 로드하여 사용


==== 사용법 ====
uv run main.py --csv data/voice_phishing_text_label.csv

uv run main.py --csv data/voice_phishing_text_label.csv --model_out models/voice_phishing_model.joblib


==== CSV 형식 예시 ====
text,label
"고객님 대출 승인되셨습니다 수수료 입금 후 송금",1
"내일 회의 자료 준비해주세요",0
"""

# LinearSVC: https://data-scientist-brian-kim.tistory.com/72#google_vignette
# - Support Vector Classifier
# - 데이터를 직선(Linear)로 나눠주는 분류기 - 2차, 3차도 가능
# - 데이터가 선형적으로 구분 가능하다면 빠르고 성능이 좋음
# - 결정경계: 데이터를 나누는 선/평면
# - 서포트벡터: 결정 경계에 가장 가까이 있는 데이터 점 -> 이 점들이 경계를 만드는 핵심
# - 마진: 두 클래스 사이의 거리, 마진이 넓을수록 일반화 성능(예측력)이 좋음

# 필요 모듈
import argparse, os  # argparse: 커맨드라인 인자 처리 / os: 운영체제 관련 기능 제공
import pandas as pd  # pandas: 데이터프레임 처리 및 데이터 분석 라이브러리
from sklearn.model_selection import train_test_split  # 데이터셋 분할 (학습/검증/테스트)
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # 텍스트 데이터를 TF-IDF 벡터로 변환
from sklearn.svm import LinearSVC  # 선형 서포트 벡터 머신(SVM) 분류기
from sklearn.pipeline import Pipeline  # 여러 처리 과정을 연결하여 파이프라인 구성
from sklearn.metrics import classification_report  # 모델 성능 평가 지표 출력
import joblib  # 학습된 모델 저장 및 로드 (직렬화)


def train(csv_path: str, model_out: str):
    """
    보이스 피싱 탐지 모델 학습
    Args:
        csv_path: 학습 데이터 파일 경로
        model_out: 학습된 모델 저장 경로
    """
    # 1. 데이터 로드 및 기본 검증 (df: data frame)
    df = pd.read_csv(csv_path)
    # print(df.head())
    # print(df.info())

    # 1-1. 데이터 전처리: 결측치 제거
    df = df.dropna(subset=["text", "label"].copy())

    # print(len(df))  # 1218

    # 1-2. text는 문자열, label은 정수형으로 변환
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # 1-3.라벨 클래스별 데이터 갯수 확인
    label_counts = df["label"].value_counts()
    # print(label_counts)  # 1-609 / 0-609

    # 2. 데이터 분할 (학습: 80%, 테스트: 20%)
    # train_test_split(입력 데이터, 타겟 데이터, 테스트 사이즈, 랜덤값 조정, 클래스 비율 유지 코드)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"].values,
    )

    # print(f"학습 데이터 개수: {len(X_train)} 개")
    # print(f"테스트 데이터 개수: {len(X_test)} 개")

    # 3. 모델 학습 파이프라인 구성
    pipeline = Pipeline(
        [
            # tfidf 벡터화
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=r"(?u)\b\w+\b",  # 한국어 처리를 위한 토큰 패턴
                    ngram_range=(1, 2),  # 1-gram, 2-gram 사용
                    max_features=10000,  # 특성을 최대 10,000개로 제한 (메모리 효율)
                    min_df=2,  # 최소 2번 이상 등장하는 단어만 사용 (노이즈 제거)
                    analyzer="word",  # 단어를 대상으로 분석
                ),
            ),
            # 선형 서포트 벡터 머신 분로
            (
                "clf",
                LinearSVC(
                    C=1.0,  # 규제 강도 (수치가 클수록 복잡도가 높아짐)
                    random_state=42,  # 랜덤 시드
                ),
            ),
        ]
    )

    # 4. 모델 학습
    pipeline.fit(X_train, y_train)

    # 5. 모델 평가
    pred = pipeline.predict(X_test)

    # 6. 성능 테스트 (정확도, 정밀도, 재현률, F1-score)
    # digits=4 : 소수점 4자리까지 표시
    print(classification_report(y_test, pred, digits=4))

    # 7. 모델 저장 디렉토리 생성
    # exist_ok=True : 디렉토리가 있으면 그냥 실행, 없으면 생성
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    # 8. 모델 저장
    joblib.dump(pipeline, model_out)
    print("=" * 50)
    print("모델이 저장되었습니다.")
    print("=" * 50)


if __name__ == "__main__":
    # 커맨드 라인 파라미터 정의
    parser = argparse.ArgumentParser(description="한국어 보이스피싱 탐지 모델 학습")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument(
        "--model_out", type=str, default="models/voice_phishing_model.joblib"
    )

    args = parser.parse_args()
    print("=" * 50)
    print(f"입력파일: {args.csv}")
    print(f"출력파일: {args.model_out}")
    print("=" * 50)

    train(args.csv, args.model_out)
