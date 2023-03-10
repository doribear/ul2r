# UL2R을 이용한 언어 모델 성능 측정 실험

## 1. 개요
<p> 
 2022년 11월 29일 구글 AI BLOG에<strong><a href = "https://ai.googleblog.com/2022/11/better-language-models-without-massive.html">'Better Language Models Without Massive Compute'</a></strong>가 포스팅 되었습니다. 해당 게시글에서는 UL2R 및 Flan
이라는 방식의 언어 모델의 성능 개선 방법을 제시 했습니다.[1] UL2R은 기존 사전학습 방법에 단순한 MLM이 아닌 다양한 noising기법을 추가하는 방식이며,
FLan은 chain of thought, few shot등 다양한 prompt를 fine-tuning과정에 추가하는 방식입니다. 구글에서는 이를 PaLM을 통해 computing power를 절약하며, 
LM을 고도화할 수 있는 방안을 제시한 것입니다. 그러나, PaLM자체도 Billion단위의 거대모델이므로 이에 대해 더 작은 모델로도 가능한지 확인하고자 상대적으로 작은 크기인
4M크기의 LM을 이용해보고자 했습니다. LM의 구조는 Albert를 이용했으며,[2] 이 저장소를 방문하시는 연구자 분들께 좋은 참고자료가 되기를 바랍니다.
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
 ALBERT는 BERT의 변형모델로 2019년 <a href="https://arxiv.org/abs/1909.11942">A Lite BERT for Self-supervised Learning of Language Representations</a>에 소개된 모델입니다. BERT와의 차이점은 모든 encoder의 parameter를 공유하는 Cross-layer parameter sharing과 입력 embedding의 크기를 줄이고 embedding과 encoder사이에 차원의 크기를 조절할 수 있도록 추가적인 linear layer를 추가하는 Factorized embedding layer parameterization이 적용된 점이 다릅니다. ALBERT는 BERT보다 상대적으로 가벼운 모델이지만 우수한 성능을 보이는 것으로 알려졌기에 ALBERT를 이용해 이번 실험을 진행했습니다.

## 5. 결과
 아래 결과를 확인해보면 작은모델에서 flan이 mlm에 비해 성능에서 더 우수하다고 결론 내릴 수는 없을것으로 보입니다. 그 이유는 사실 mlm을 이용해 pretrain을 한 결과는 사실상 제대로 된 pretrain이 이루어 지지 않은 것으로 보이기에 최종 fine tuning결과의 성능을 비교할 수 없기 때문입니다. 하지만, flan과 mlm두 pretrain에 대해 동일한 데이터로 동일한 횟 수 학습을 했음에도 불구하고, flan에서는 적절한 pretrain이 이루어진 것을 확인할 수 있기에 ul2r을 상대적으로 작은 모델에서도 시도해 볼만 하다고 결론 내릴 수 있습니다.
<img src="mlm_flan비교.png">

## 6. Reference
[1] Better Language Models Without Massive Compute https://ai.googleblog.com/2022/11/better-language-models-without-massive.html<br>
[2] Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2019). Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.
