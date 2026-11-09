## 5단계 결과 처리 + 1~5단계 통합

## 실행 방법 (노트북)

`5-result` 폴더에서 순서대로 더블클릭

1. `setup.bat` : 처음 한 번 설치 (Python 3.12 이상)
2. `enroll.bat` : 이름 입력 후 정면을 보며 얼굴 등록
3. `run.bat` : 통합 실행 (등록자 `OPENED`, 미등록 `RETRY` 3번 후 `LOCKED`, 종료 ESC)

## 실행 방법 (라즈베리파이)

```bash
pip install -r 5-result/requirements.txt
python 5-result/stages/models.py
python 5-result/integrate.py enroll --name 이름
python 5-result/integrate.py run
```
핀 번호는 `set5.json`에서 배선에 맞게 수정

## 시행착오

1. **problem** 버튼이 없을 때 엔터로 대신하는 기능이, 입력이 없으면 눌린 걸로 처리되어 문이 열림

   **solution** 엔터 대체는 --demo에서만 켜고, 버튼이 없으면 무조건 잠금

2. **problem** 앞사람의 실패 횟수가 다음 사람에게 넘어감

   **solution** 마지막 실패 후 30초(session_timeout)가 지나면 실패 횟수 초기화

3. **problem** 경보음이 울리는 동안 누른 버튼은 무시됨

   **solution** 경보 시작부터 버튼 입력을 받도록 변경

4. **problem** 1단계·4단계 코드가 바뀔 때마다 5단계 연결 코드가 멈춤

   **solution** 4단계 결과 (이름, 점수)만 받도록 분리, 이름이 없거나 값이 이상하면 문을 열지 않음

5. **problem** 1단계와 5단계가 LED를 각자 만들면 같은 핀이 겹침

   **solution** 5단계에서 RGBLED 하나만 만들어 1단계에도 넘겨 같이 사용

6. **problem** 임시통합은 1~4단계만 묶여 있고, 인증 실패는 기록 없이 계속 대기함

   **solution** 임시통합 1~4단계 모듈을 stages/로 복사하고 integrate.py에서 4단계 결과 (이름, 점수)를 5단계로 넘김.
   실패도 로그·사진으로 남기고, 3회 실패 시 경보·수동 버튼·잠금 처리
