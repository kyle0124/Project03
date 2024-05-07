# Project03
## 중고차 판매 견적
-- 중고차 시장 규모가 커지고, 직거래 시장이 활성화되는 추세에서 개인이 판매자로서 중고차 시세 정보를 간편하게 파악할 수 있도록 제작

-- 차량 기본 정보 및 옵션 입력 시 적정 판매가 제공
  - 마진율을 고려한 적정 판매가
  - 80여가지의 옵션을 활용한 신뢰도 높은 가격 예측 (feature importance가 가장 높은 10가지 옵션 위주로 제공)
  - 차량별 최적화된 예측모델 사용 (Pycaret을 이용하여 차량 별 가장 성능이 좋은 모델 적용)

- 개발환경 : Windows 10
- 개발도구 : Jupyter Notebook, Google Colab, VisualStudio Code
- 개발언어 : Python
- Data : 중고차 판매 사이트 Crawling
- 사용 기술(라이브러리) : scikit-learn, ensemble, voting, pycaret, streamlit
- 웹 어플리케이션 주소 : https://predict-my-car.streamlit.app/
