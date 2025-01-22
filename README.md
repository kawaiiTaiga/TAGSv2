# TAGS: 텍스트 증강 프레임워크

> 생성과 선정을 통한 효과적인 텍스트 증강 방법론

## 소개
TAGS(Text Augmentation with Generation and Selection)는 거대 언어 모델과 퓨샷 러닝을 활용하여 텍스트 데이터를 증강하는 프레임워크입니다. 적은 양의 원본 데이터로도 높은 품질의 증강 텍스트를 생성할 수 있습니다.

## 주요 특징
* 퓨샷 러닝 기반의 다양한 텍스트 생성
* 대조 학습을 통한 스마트한 텍스트 선정
* 반복적 증강으로 데이터 확장
* 클래스 정보 보존

## 설치 방법
```bash
git clone https://github.com/KawaiiTaiga/TAGS.git
cd TAGS
pip install -r requirements.txt
```

## 필요 라이브러리
* Python 3.x
* PyTorch
* Transformers
* KoGPT
* KcBERT

## 성능
* 데이터 증강: 60배 이상 증가
* 모델 성능: 0.1915+ 향상
* 의미론적, 표현적 다양성 확보

## 논문 정보
```bibtex
@article{TAGS2023,
    title={TAGS: Text Augmentation with Generation and Selection},
    author={김경민 and 김동환 and 조성웅 and 오흥선 and 황명하},
    journal={정보처리학회논문지/소프트웨어 및 데이터 공학},
    volume={12},
    number={10},
    pages={455-460},
    year={2023}
}
```

## 라이선스
이 프로젝트는 Creative Commons Attribution Non-Commercial 라이선스를 따릅니다.

