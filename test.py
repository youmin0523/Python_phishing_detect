import joblib

model = joblib.load("models/voice_phishing_model.joblib")


def predict_text(model, text):
    """
    텍스트를 입력받아 피싱 여부 예측
    Args:
      model: 학습된 모델
      text: 예측할 텍스트
    Return:
      tuple: (예측 결과(str), 예측 확률(float), 신뢰도, 원본 점수)
    """

    prediction = model.predict([text])[
        0
    ]  # predict는 2차원 배열을 반환하기 때문에 첫번째 요소 선택
    decision_score = model.decision_function([text])[0]  # decision_score 예시값: 0.567
    confidence = abs(decision_score)

    # 타겟값에 따른 필터링과 신뢰도 실수에 따른 필터링
    if prediction == 1:
        result = "보이스피싱"
        # 양수 점수면 더해서 1.0에 가깝게
        confidence_percent = (
            min(0.5 + (decision_score / 10), 1.0) if decision_score > 0 else 0.5
        )
    else:
        result = "정상"
        # 음수 점수면 빼서(마이너스 마이너스는 플러스) 1.0에 가깝게
        confidence_percent = (
            min(0.5 - (decision_score / 10), 1.0) if decision_score < 0 else 0.5
        )

    return result, confidence_percent, decision_score


def batch_test_examples(model):
    """
    미리 정의된 텍스트 테스트
    신뢰도 해석:
      0.5 미만 정상 텍스트 확률 높음
      0.5 이상이면 피싱 텍스트 확률 높음
    """
    examples = [
        "안녕하세요, 내일 회의가 있습니다.",
        "고객님 대출이 승인되었습니다. 수수료를 입금해주세요.",
        "검찰청입니다. 명의도용 사건으로 연락드렸습니다.",
        "오늘 저녁에 치킨 드실래요?",
        "신용카드 한도가 증액되었습니다. 확인을 위해 비밀번호를 알려주세요.",
        "금융감독원에서 연락드렸습니다. 계좌 확인이 필요합니다.",
        "친구야, 언제 만날까?",
        "통장 발급을 위해 개인정보가 필요합니다.",
    ]

    for i, text in enumerate(examples, 1):
        result, confidence, decision_score = predict_text(model, text)
        print("-" * 100)
        print(f"{i}번째 텍스트: {text}")
        print(f"예측 결과: {result}")
        print(f"신뢰도: {confidence:.4f}")
        print(f"원본 점수: {decision_score:.4f}")
        print("-" * 100)


def interactive_mode(model):
    """대화형 모드 테스트"""
    while True:
        text = input("텍스트를 입력해주세요: ").strip()

        if text.lower() in ["quit", "exit", "종료"]:
            print("테스트를 종료합니다.")
            break

        if not text:
            print("텍스트를 입력해주세요.")
            continue

        result, confidence, decision_score = predict_text(model, text)
        print("-" * 100)
        print(f"예측 결과: {result}")
        print(f"신뢰도: {confidence:.4f}")
        print(f"원본 점수: {decision_score:.4f}")
        print("-" * 100)


if __name__ == "__main__":
    batch_test_examples(model)
    print("\n" + "=" * 20 + " 대화형 모드 시작 " + "=" * 20)
    interactive_mode(model)
