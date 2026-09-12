# quickstart 번들

데이터셋 다운로드나 학습 없이 툴킷 전체 흐름을 40초 안에 돌리기 위한 파일들입니다. 저장소 루트에서 `make quickstart`
(또는 README 상단 "3분 안에 보기"의 세 명령)로 씁니다.

| 파일 | 내용 | 출처 |
|---|---|---|
| `resnet20_relu.pt` | ResNet-20 ReLU, CIFAR-10 검증 분할로 고른 체크포인트 (테스트 89.59%) | `runs/resnet20_relu/best.pt`, E1 베이스라인과 동일 |
| `cust_inception.pt` | 가상 고객 B의 Inception-32 (concat 분기 CNN, 테스트 89.59%) | `runs/cust_inception/best.pt`, E9와 동일 |
| `cifar10_sample.npz` | CIFAR-10에서 seed 0으로 뽑은 학습 512장(캘리브레이션용) + 테스트 500장, uint8, `val_per_class = 0` | Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009 |

샘플 500장으로 잰 정확도는 전체 10,000장 결과와 ±1.5%p 정도 다를 수 있습니다(표준오차 ≈ 1.4%p). 본문 표의 숫자는 모두
전체 테스트 분할로 잰 `results/*.json`의 값입니다.
