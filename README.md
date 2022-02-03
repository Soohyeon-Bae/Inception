EDIT : Reproduced from Kangmin Park's 'Inception.py' & 'utils.py'

# Inception

[CVPR 2015 Open Access Repository](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)

# 코드 구현

- `num_of_classes`랑 `units`를 모두 2로 바꿨는데 어디서 shape이 102로 나오는지 모르겠음
- `BinaryCrossentropy()`로 오차함수 변경

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d5312135-4d00-4945-9973-0eee041c38bb/Untitled.png)

[https://github.com/Soohyeon-Bae/Inception](https://github.com/Soohyeon-Bae/Inception)

# 논문 리뷰

# Motivation and High Level Considerations

- 심층 신경망의 성능을 향상시키기 위해 네트워크의 depth와 width를 크게 하는 방법이 있지만, 다음의 단점이 존재함
1. 학습할 파라미터가 많아지면, 학습 데이터가 적은 경우 오버피팅이 일어나기 쉽다.
2. 컴퓨터 자원의 사용이 증가한다.

> 위 문제를 해결하기 위한 방법은 fully-connected를 sparsely-connected 아키텍쳐로 변경하는 것이다.
> 

# Architectural Details

> 최적의 local sparse structure를 구성하고 어떻게 dense components를 구성할 수 있을지
> 
1. 성능 향상을 위해 네트워크를 늘리면 오버피팅과 계산량이 크게 증가함 → 네트워크를 **sparse**하게 구성하여 크기를 증가시킴
2. sparse한 구성은 하드웨어 연산에 비효율적임 →  sparse structure를 **dense** components로 구성함

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0f5c1e16-98b0-425f-9dd8-77e698bef120/Untitled.png)

Fig.2(a) : 작은 크기의 conv 레이어라도 많이 쌓이면 연산량이 많아짐

**Fig.2(b)** : **1x1 conv 레이어**를 통한 차원 축소로 연산량 감소

연산량을 크게 늘리지 않으면서 네트워크를 키울 수 있음

![(왼) Sparse (오) Dense](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/672cd846-29cc-4af7-a132-e329bb349969/Untitled.png)

(왼) Sparse (오) Dense

- Sparse 매트릭스를 서로 묶어(clustering) 상대적으로 dense한 서브 매트릭스를 만듦

# GoogleNet

- 입력 이미지의 크기는 224 x 224로 RGB 컬러 채널을 가지며, mean subtraction 적용하여 전처리
- 22개의 층으로 이루어져 있음
- 모든  conv 레이어에 ReLU 적용
- **#3x3 reduce,** **#5x5 reduce :** 3x3과 5x5 conv 레이어 앞에 사용되는 1x1 필터의 채널 수
- **pool proj** : maxpooling layer 뒤에 오는 1 x 1 필터의 채널 수

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6f0ead74-3f9e-401d-9f8d-403abdf06776/Untitled.png)

- 메모리 효율성을 위해 처음에는 기본적인 CNN 구조 사용

### 1x1 Convolution Layer

지난 VGG에서와 마찬가지로, 1x1 conv 연산의 수행은 다음의 장점을 가짐.

1. 차원 축소를 통한 연산량 감소
2. 활성화 함수(ReLU) 포함으로 비선형성 강화

### Concatenation

- 이전 층에서 생성된 특성맵을 1x1 conv, 3x3 conv, 5x5 conv, 3x3 maxpooling 한 다음, 얻은 특성맵들을 모두 쌓아 다양한 특성이 도출되도록 함
- 이에 1x1 conv 레이어를 포함시켜 연산량을 줄임

### Global Average Pooling

> 전 층에서 산출된 특성맵들을 각각 평균낸 것을 이어서 1차원 벡터를 만들어주는 것이다. 1차원 벡터를 만들어줘야 최종적으로 이미지 분류를 위한 softmax 층을 연결해줄 수 있기 때문이다.
> 

> 만약 전 층에서 1024장의 7 x 7의 특성맵이 생성되었다면, 1024장의 7 x 7 특성맵 각각 평균내주어 얻은 1024개의 값을 하나의 벡터로 연결해주는 것이다.
> 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ffb6d8c4-2f80-4726-a880-9a6c64d30d53/Untitled.png)

- FC 레이어에서는 위와 같은 연산을 수행할 때 7x7x1024x1024개의 가중치가 필요하지만,GAP을 사용하면 가중치가 한 개도 필요하지 않음

### A**uxiliary Classifier**

> 네트워크의 깊이가 깊어지면 깊어질수록 vanishing gradient 문제를 피하기 어려워진다. 이 문제를 극복하기 위해서 네트워크 중간에 두 개의 보조 분류기(auxiliary classifier)를 달아주었다.
> 
- vanishing gradient
    
    가중치를 훈련하는 과정에 역전파(back propagation)를 주로 활용하는데, 역전파 과정에서 가중치를 업데이트하는데 사용되는 gradient가 점점 작아져서 0이 되어버리는 것이다. 따라서 네트워크 내의 가중치들이 제대로 훈련되지 않는다. 
    
- 신경망 중간에 예측 결과를 출력함
- 적절하게 역전파가 적용될 수 있도록 하기 위함
- 보조 분류기에서 계산된 error는 0.3 가중치를 곱하여 최종 error에 더해짐
- 훈련 시에만 사용

![075E24AA-7A48-4DBC-8ECF-91FD31FC0AB1.jpeg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f11a9771-7dfa-4682-b60b-48ee76f4c220/075E24AA-7A48-4DBC-8ECF-91FD31FC0AB1.jpeg)

- 참고
    
    [구글 인셉션 Google Inception(GoogLeNet) 알아보기](https://ikkison.tistory.com/86)
    
    [[CNN 알고리즘들] GoogLeNet(inception v1)의 구조](https://bskyvision.com/539)
    
    [[논문] GoogleNet - Inception 리뷰 : Going deeper with convolutions](https://leedakyeong.tistory.com/entry/%EB%85%BC%EB%AC%B8-GoogleNet-Inception-%EB%A6%AC%EB%B7%B0-Going-deeper-with-convolutions-1)
