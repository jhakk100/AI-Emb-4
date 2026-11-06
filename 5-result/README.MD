## 시행착오

1. **problem** 4단계 `authenticate_face()`는 성공/실패 값 없이 `(이름 또는 None, 점수)`만 돌려주고,
   등록된 사람이 없으면 `(None, None)`이 들어옴

   **solution** `from_stage4()`에서 이름이 있으면 성공, None이면 실패로 처리(등록자 없음은 점수 -1로 기록).
   형식이 맞지 않는 값은 문을 열지 않고 오류를 냄(보안 시스템이라 실패하면 닫히는 쪽)

2. **problem** 1~4단계에는 LED 핀 설정이 없고, 1단계는 `prepare_camera(led=...)`로 LED 객체만 받음.
   1단계와 5단계가 LED를 각자 만들면 같은 핀을 두 번 열게 됨

   **solution** set5.json에 핀을 두고 `make_devices()`로 RGBLED를 하나만 만들어 1단계에도 넘김.
   흰색 깜빡임(대기) -> 노랑(준비) -> 초록(성공)/빨강(실패) 순서로 같이 사용

3. **problem** 1단계는 준비 완료 후 카메라를 닫고 프레임을 버려서, 결과 사진(과제 조건)을 남길 프레임이 없음

   **solution** 인증에 쓴 프레임을 `from_stage4(결과, frame=...)`로 받아서 logs/snapshots/에 저장
