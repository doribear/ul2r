# UL2R을 이용한 언어 모델 성능 측정 실험

## 1. 개요
<p> 
 2022년 11월 29일 구글 AI BLOG에<strong><a href = "https://ai.googleblog.com/2022/11/better-language-models-without-massive.html">'Better Language Models Without Massive Compute'</a></strong>가 포스팅 되었습니다. 해당 게시글에서는 UL2R 및 Flan
이라는 방식의 언어 모델의 성능 개선 방법을 제시 했습니다. UL2R은 기존 사전학습 방법에 단순한 MLM이 아닌 다양한 noising기법을 추가하는 방식이며,
FLan은 chain of thought, few shot등 다양한 prompt를 fine-tuning과정에 추가하는 방식입니다. 구글에서는 이를 PaLM을 통해 computing power를 절약하며, 
LM을 고도화할 수 있는 방안을 제시한 것입니다. 그러나, PaLM자체도 Billion단위의 거대모델이므로 이에 대해 더 작은 모델로도 가능한지 확인하고자 상대적으로 작은 크기인
4M크기의 LM을 이용해보고자 했습니다. LM의 구조는 Albert를 이용했으며, 이 저장소를 방문하시는 연구자 분들께 참고자료가 되기를 바랍니다.
</p>
<p>
 Flan의 경우에는 human power를 통해 추가적인 데이터를 생성해야하는 문제가 있어 해당 포스팅에서 제시된 내용 중 UL2R만을 적용했습니다. 또한 MLM과의 직접적 비교를 위해
MLM vs UL2R에 대해서만 비교했습니다.(사실 컴퓨팅 파워의 한계로 가급적 비교절차를 간소화 하려고 한 게 더 큰 이유...) 
</p>

## 2. 데이터 셋
 AI-HUB의 <strong><a href = "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86">감성대화 말뭉치</a></strong>
를 이용했습니다. 해당 데이터에 대해서는 AI-HUB를 참고해 주세요

## 3. UL2R
<p>
UL2R은 MLM과 유사하게 masking된 token을 복원하는 과제인데, UL2R에서는 3가지 유형이 존재합니다. 
<table>
<tr>
<th>noising</th>
<th>noising 방식</th>
</tr>
<tf>
<td>r-denoiser</td>
<td>문장의 15%를 noise로 사용</td>
</tr>
<tf>
<td>s-denoiser</td>
<td>noise의 수가 많거나 긴 경우로 noise가 보통 32개의 corruption이거나 50%이상</td>
</tr>
<tf>
<td>x-denoiser</td>
<td>PrefixLM objective라고 알려진 방법으로 문장의 랜덤한 시작점부터 랜덤한 지점까지를 noising하는 방법</td>
</tr>
</table>

구글은 PaLM에 UL2R을 적용해 U-PaLM이라고 이름을 붙였는데 이 U-PaLM에는 S-Denoiser 50%, R-denoiser 25%, X-Denoiser 25%가 적용 되었습니다.
</p>
 <p>
 전통적으로, 대부분의 LM은 NSP나 MLM과 같은 방법으로 pre-training을 진행했습니다. 비록 NSP와 같은 causal language modeling이 long-form generation에서 유리하고, 
 MLM과 같은 denoising objective가 fine-tuning에 유리하다는 trade-off관계가 성립하지만, google ai team은 mixture-of-denoisers objective를 
 사용하는 것이 두 가지 시나리오 에서 모두 기존의 방식보다 더 잘 작동함을 보여주기도 했습니다.
 </p>
<p>
 아래는 실제로 UL2R을 적용한 PaLM의 결과인데 UL2R을 적용하는 것이 상당히 좋은 결과를 보여주는 것은 확실해 보입니다.
</p>
<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhPrW5Qz2au3fkXwUS8eUUqoP9Afd6Gl7pJPHjGVSPZpwy-7hMwMzKMNOigdEeWJgpQe8ODTTMzAE3h-_BZAbiYIxRqvEj3IDlKXHZpf3INnFx37jJFqIUIO3Ug0HStDtgEVhaugX7WeQowEAiTPVuez3dTwu-A-VpdvmQbmEtUSWrb8_hMy6-sgEnPVw/s16000/BigBenchPerform.png">

## 4. ALBERT

## 5. 결과

## 6. Reference
