# Factor Graph-based Interpretable Neural Networks
## Abstract
Comprehensible neural network explanations are foundations for a better understanding of decisions, especially when the input data are infused with malicious perturbations. Existing solutions generally mitigate the impact of perturbations through adversarial training, yet they fail to generate comprehensible explanations under unknown perturbations. To address this challenge, we propose AGAIN, a fActor GrAph-based Interpretable neural Network, which is capable of generating comprehensible explanations under unknown perturbations. Instead of retraining like previous solutions, the proposed AGAIN directly integrates logical rules by which logical errors in explanations are identified and rectified during inference. Specifically, we construct the factor graph to express logical rules between explanations and categories. By treating logical rules as exogenous knowledge, AGAIN can identify incomprehensible explanations that violate real-world logic. Furthermore, we propose an interactive intervention switch strategy rectifying explanations based on the logical guidance from the factor graph without learning perturbations, which breaks the inherent limitation of adversarial training-based methods in defending only against known perturbations. Additionally, we theoretically demonstrate the effectiveness of employing factor graph by proving that the comprehensibility of explanations is strongly correlated with factor graph. Extensive experiments are conducted on three datasets and experimental results illustrate the superior performance of AGAIN compared to state-of-the-art baselines.
![teaser](https://github.com/yewsiang/ConceptBottleneck/blob/master/figures/tti_qual_examples.png)
## Prerequisites
For CBM:
- matplotlib 3.1.1
- numpy 1.17.1
- pandas 0.25.1
- Pillow 6.2.2
- scipy 1.3.1
- scikit-learn 0.21.3
- torch 1.1.0
- torchvision 0.4.0

For Factor Graph:
- pracmln 1.2.2
