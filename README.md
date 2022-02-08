# Inception

[CVPR 2015 Open Access Repository](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)

# 코드 구현

[https://github.com/Soohyeon-Bae/Inception](https://github.com/Soohyeon-Bae/Inception)

# 논문 리뷰

# Motivation and High Level Considerations

- 심층 신경망의 성능을 향상시키기 위해 네트워크의 depth와 width를 크게 하는 방법이 있지만, 다음의 단점이 존재함
1. 학습할 파라미터가 많아지면, 학습 데이터가 적은 경우 오버피팅이 일어나기 쉽다.
2. 컴퓨터 자원의 사용이 증가한다.

> 위 문제를 해결하기 위한 방법은 convolution 과정 안에서 fully-connected를 sparsely-connected 아키텍쳐로 변경하는 것이다.
> 

# Architectural Details

> 최적의 local sparse structure를 구성하고 어떻게 dense components를 구성할 수 있을지
> 

### **Translation Invariance 가정**

- Input의 위치가 달라져도 output이 동일한 값을 갖는다.
    
    ![**CNN 네트워크 자체는 translation equivariance(variance), feature의 위치가 바뀌면 당연히 output에서 해당 feature에 대한 연산결과의 위치도 바뀌기 때문**](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ff02149d-65be-4bee-a408-ccea213cbba8/Untitled.png)
    
    다음의 과정을 통해 CNN이 translation invariance하게 만들 수 있다.
    
    1. Max pooling
        
        > max-pooling은 대표적인 **small translation invariance**함수이다. 예를 들어 orginal image의 pixel값이 [1, 0, 0, 0]일 때, original image를 각각 translate시킨 A, B는 [0, 0, 0, 1], [0, 1, 0, 0]의 값을 갖고 있다. 이를 2x2 max pooling시키면 모두 동일하게 output으로 1을 갖는다.
        > 
        
        > Max pooling은 k x k filter 사이즈만큼의 값들을 1개의 max값을 치환시킨다. 따라서 k x k 내에서 값들의 위치가 달라져도 모두 동일한 값을 output으로 갖게 된다.
        > 
        
        **"즉, k x k 범위내에서의 translation에 대해서는 invariance하다."**
        
    2. Weight sharing & Learn local features
        
        > Weight sharing : 동일한 weight를 가진 filter를 sliding window로 연산한다.
        > 
        
        > Learn local features : global이 아닌 local feature들과 연산함으로써 local feature를 학습한다.
        > 
        
        ![동일 가중치를 모든 픽셀이 공유하면서 local하게 연산하기에 FC layer에서의 output값들도 input image의 local value들의 영향을 받아 특정 사이즈 내에서만 equivariant하게 값이 바뀐다](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/44ec2cfc-39d7-4704-8180-cd099da8eedc/Untitled.png)
        
        
        **각 필터는** image내 어떤 **object의 위치와 상관없이 특정 패턴을 학습**하는 것
        
    3. Softmax
        
        ![classification으로 output k가 나오기 위해서는 feature map k에 높은 값이 많으면 되는 것이고 feature map k에 높은 값이 많으려면 channel k와 original image가 비슷한 패턴을 갖고 있어야 한다.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/97b12dad-6591-4681-919d-05793c1950ff/Untitled.png)
        
        
        **"즉, object의 위치와 상관없이 패턴이 동일하면 동일한 output을 갖게 된다"**
        

1. 성능 향상을 위해 네트워크를 늘리면 오버피팅과 계산량이 크게 증가함 → 네트워크를 **sparse**하게 구성하여 크기를 증가시킴
2. sparse한 구성은 하드웨어 연산에 비효율적임 →  필터 수준의 sparsity을 사용하지만 dense matrices에 대한 계산으로 하드웨어를 활용하는 아키텍처(각각의 연산은 sparse하지만 concat해서 묶고 보니 dense하게 되었다.)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0f5c1e16-98b0-425f-9dd8-77e698bef120/Untitled.png)

receptive fieldFig.2(a) : 작은 크기의 conv 레이어라도 많이 쌓이면 연산량이 많아짐

**Fig.2(b)** : **1x1 conv 레이어**를 통한 차원 축소로 연산량 감소

연산량을 크게 늘리지 않으면서 네트워크를 키울 수 있음

![(왼) Sparse (오) Dense](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/672cd846-29cc-4af7-a132-e329bb349969/Untitled.png)


- 성능 향상을 위해 sparse 매트릭스를 서로 묶어(clustering) 상대적으로 dense한 서브 매트릭스를 만듦

# GoogleNet

> The size of the receptive field in our network is 224×224 taking RGB color channels with mean subtraction.
> 

→ 여기서 receptive field는 ‘수용영역’이 아니라 ‘연산이 수행될 공간의 크기’ 정도로 해석하는 게 맞을 것 같음.

- 수용영역(receptive field)
    
    > 출력 레이어의 뉴런 하나에 영향을 미치는 입력 뉴런들의 공간 크기
    > 
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4146c5d7-fd67-4de4-a71a-2705e172d27b/Untitled.png)
    
- 파라미터가 존재하는 레이어는 22개층이고, pooling을 포함하면 27개층으로 구성
- 모든  conv 레이어에 ReLU 적용
- **#3x3 reduce,** **#5x5 reduce :** 3x3과 5x5 conv 레이어 앞에 사용되는 1x1 필터의 채널 수
- **pool proj** : maxpooling layer 뒤에 오는 1 x 1 필터의 채널 수

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6f0ead74-3f9e-401d-9f8d-403abdf06776/Untitled.png)

- 메모리 효율성을 위해 처음에는 기본적인 CNN 구조 사용

### 1x1 Convolution Layer

지난 VGG에서와 마찬가지로, 1x1 conv 연산의 수행은 다음의 장점을 가짐.

1. 차원 축소를 통한 연산량 감소
    - 1x1 conv 필터의 개수가 input dimension보다 작다면 차원 축소가 일어나게 됨
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f7ee3982-c162-4ed3-8ffa-0ece85136ed7/Untitled.png)
        
        ![1x1필터를 32개 깔면 output은 spatial dimension은 유지하면서 depth는 32개로 줄어듦](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/16c68c9b-f676-4003-bebe-60b7425d6065/Untitled.png)
        
        
2. 활성화 함수(ReLU) 포함으로 비선형성 강화

### Concatenation

- 이전 층에서 생성된 특성맵을 1x1 conv, 3x3 conv, 5x5 conv, 3x3 maxpooling 한 다음, 얻은 특성맵들을 모두 쌓아 다양한 특성이 도출되도록 함
- 이에 1x1 conv 레이어를 포함시켜 연산량을 줄임
- max-pooling
    
    > 각 window에서 가장 큰 자극만을 선택하는 것
    > 
    - Input의 위치가 달라져도 output이 동일한 값을 갖도록 하기 위해서 (Translation Invariance)
- pooling과 convolution 연산 차이점
    - Convolution의 크기는 연산 필터의 크기
    - Pooling의 크기는 출력의 크기

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
    
- 신경망 중간에 예측 결과를 출력하여 적절하게 역전파가 적용될 수 있도록 하기 위함
- 보조 분류기에서 계산된 error는 0.3 가중치를 곱하여 최종 error에 더해짐
- 훈련 시에만 사용
1. 5x5 average pooling(stride 3)을 거쳐 4x4x512 또는 4x4x528 출력
2. 1x1 conv(128 filters) 레이어를 거쳐 차원을 줄이고  ReLU 적용
3. 1024 유닛을 가진 FC 레이어를 지남
4. 드롭아웃 70% 적용
5. 소프트맥스를 통해 1000개 클래스에 대한 분류 

![075E24AA-7A48-4DBC-8ECF-91FD31FC0AB1.jpeg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f11a9771-7dfa-4682-b60b-48ee76f4c220/075E24AA-7A48-4DBC-8ECF-91FD31FC0AB1.jpeg)

# Conclusions

> Moving to **sparser** architectures is feasible and useful idea in general.
> 

- 참고
    
    [구글 인셉션 Google Inception(GoogLeNet) 알아보기](https://ikkison.tistory.com/86)
    
    [[CNN 알고리즘들] GoogLeNet(inception v1)의 구조](https://bskyvision.com/539)
    
    [[논문] GoogleNet - Inception 리뷰 : Going deeper with convolutions](https://leedakyeong.tistory.com/entry/%EB%85%BC%EB%AC%B8-GoogleNet-Inception-%EB%A6%AC%EB%B7%B0-Going-deeper-with-convolutions-1)
    
    [7. CNN 구조 2 GoogleNet](https://m.blog.naver.com/laonple/221229726012)
    
    [[DL] 1x1 convolution은 무엇이고 왜 사용할까?](https://euneestella.github.io/research/2021-10-14-why-we-use-1x1-convolution-at-deep-learning/)
    
    [[cs231n] 9 강. CNN architectures](https://bookandmed.tistory.com/58)
    
    [[Image Classification] GoogleNet(Inception V1) : Going Deeper with Convolutions](https://kkkkhd.tistory.com/10)
    
    [translation invariance 설명 및 정리](https://ganghee-lee.tistory.com/43)
