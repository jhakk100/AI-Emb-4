Threshold의 중요성 확인
- 얼굴 유사도가 일정 기준 이상이면 인증하는 구조임을 확인
- Threshold가 낮으면 다른 사람이 잘못 통과할 가능성(FAR)이 증가
- Threshold가 높으면 본인이 거부될 가능성(FRR)이 증가
- 적절한 Threshold를 찾기 위해 여러 값을 비교할 필요가 있다고 판단

FAR / FRR 성능평가 코드 구현
- 정상 사용자(Genuine)와 타인(Impostor)의 similarity 데이터를 분리
- Threshold별 FAR과 FRR을 계산하도록 구현
- 여러 Threshold를 비교하여 성능 변화를 확인

EER 추가
- FAR과 FRR이 가장 가까워지는 지점을 찾도록 EER 계산 기능 구현
- 단순 인증 성공/실패가 아닌 오류율을 기반으로 Threshold를 평가할 수 있도록 개선

실제 데이터가 없는 문제
- 초기에는 실제 얼굴 등록 및 similarity 데이터가 없어 가상 데이터로 코드 검증
- 가상 데이터로 FAR / FRR / EER 계산 기능과 그래프 출력 기능을 먼저 테스트
- 실제 얼굴인식 데이터를 확보한 후 실제 성능평가에 적용할 예정

최종 목표
- 초기 Threshold 설정 → 실제 얼굴인식 데이터 수집
- Threshold별 FAR / FRR 비교
- EER을 참고하여 적절한 Threshold 선정
- 선정된 Threshold를 실제 얼굴인식 시스템에 반영
