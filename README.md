# Gastrointestinal-Classification
#### Aluno: Gledson Melotti (https://github.com/gledsonmelotti/Gastrointestinal-Classification).
#### Orientadora: Manoela Kohler (https://github.com/manoelakohler).
---

Trabalho apresentado ao curso de Especialização: "Visão Computacional: Interpretando o Mundo Através de Imagens - Computer Vision Master" em nível de Pós-Graduação "Lato Sensu" (https://ica.ele.puc-rio.br/cursos/computer-vision-master/) como pré-requisito para conclusão de curso.

---

## Resumo

O reconhecimento automatizado de trato gastrointestinal a partir de imagens endoscópicas têm se tornado uma área promissora com o avanço dos algoritmos de inteligência artificial (IA), como redes neurais artificiais são algoritmos que tem apresentado resultados satisfatórios na distinção entre imagens com e sem lesões ou doenças no trato gastrointestinal, identificando padrões visuais complexos que às vezes não são reconhecidos pela percepção médica tradicional. As aplicações dessas técnicas contribuíram para diagnósticos mais rápidos, padronizados e confiáveis, especialmente em contextos clínicos com alta demanda ou escassez de especialistas. Assim, o uso de IA na classificação de imagens gastrointestinais representa um importante avanço para o diagnóstico precoce e a melhoria do prognóstico de doenças do trato digestivo. Dito de outro modo, os resultados de tais algoritmos facilitam intervenções médicas para um tratamento específico. Assim, com o objetivo de contribuir para a melhoria dos resultados de classificação de imagens dos tratos gastrointestinais, este trabalho propôs estratégias de fusões para melhorar os resultados individuais dos algoritmos de inteligência artificial. Os resultados foram avaliados pelas métricas acurácias, sensibilidade, especificidade, precisão, taxa de falsos positivos, F-escore, coeficiente de correlação de Matthews, Kappa e áreas sob as curvas da Característica de Operação do Receptor e da Precisão-Sensibilidade.

## Abstract

Automated recognition of the gastrointestinal tract from endoscopic images has become a promising area with the advancement of artificial intelligence (AI) algorithms, as artificial neural networks are algorithms that have shown satisfactory results in distinguishing between images with and without lesions or diseases in the gastrointestinal tract, identifying complex visual patterns that are sometimes not recognized by traditional medical perception. The applications of these techniques have contributed to faster, more standardized, and more reliable diagnoses, especially in clinical contexts with high demand or a shortage of specialists. Thus, the use of AI in the classification of gastrointestinal images represents an important advance for the early diagnosis and improved prognosis of diseases of the digestive tract. In other words, the results of such algorithms facilitate medical interventions for specific treatment. Therefore, with the aim of contributing to the improvement of gastrointestinal image classification results, this work proposed fusion strategies to improve the individual results of artificial intelligence algorithms. The results were evaluated using the metrics accuracy, sensitivity, specificity, precision, false positive rate, F-score, Matthews correlation coefficient, Kappa, and the areas under the Receiver Operating Characteristic and Precision-Recall curves.

## 1.0 Introdução e Justificativa

O estilo de vida moderno e hábitos alimentares inadequados têm contribuído para o aumento das infecções e doenças gastrointestinais. Úlceras, pólipos, inflamações, cânceres no esôfago, estômago e cólon representam uma parcela significativa dos novos diagnósticos e mortes anuais, frequentemente associados a condições como úlceras, sangramentos e pólipos. A detecção precoce dessas infecções e doenças é possível com observações de especialistas, mas pequenos tratos costumam passar despercebidos em exames iniciais, muitas vezes devido às limitações dos métodos endoscópicos tradicionais e erros humanos [1-3].

Com o avanço da inteligência artificial (IA), os algoritmos de aprendizado de máquinas, como as redes neurais convolucionais (CNN do inglês “Convolutional Neural Networks”) [4-6], têm demonstrado alto desempenho na análise de imagens médicas [7-11]. Tais algoritmos contribuíram com o desenvolvimento de soluções acessíveis e reprodutíveis, com a finalidade de auxiliar endoscopistas na identificação de anomalias com maior precisão [12-13], que contribuiu para diagnósticos mais precoces, tratamentos mais eficazes, redução de custos e alívio da pressão sobre os sistemas de saúde [14-16], especialmente diante da projeção de aumento global de doenças gastrointestinais [17-18].

Além de aumentar a confiabilidade dos diagnósticos, as técnicas de inteligência artificial permitem o desenvolvimento de sistemas assistivos que podem auxiliar profissionais da saúde durante procedimentos clínicos ou triagens em ambientes com escassez de especialistas, bem como reduzir o tempo de análise das imagens gastrointestinais e contribuindo para o reconhecimento precoce de doenças gastrointestinais e, consequentemente, para melhores prognósticos e planos terapêuticos [7-18].

## 2.0 Objetivos

### 2.1  Objetivo geral

Este projeto tem como finalidade aperfeiçoar os resultados individuais obtidos das predições de classificações de doenças gastrointestinais [19], a utilizar estratégias de fusões intermediária (intermediate fusion) [20-22] e fusões tardias/posteriores (late fusion) [23-26].

### 2.2 Objetivos específicos

- Estudar as predições individuais obtidas a partir das redes neurais convolucionais EfficientNetV2-Medium [27] e ConvNeXtV2-Large [28], bem como a rede neural transformadora DinoV2 [29].
- Apresentar estratégias de fusões tardias usando as predições individuais das classificações da EfficientNetV2-Medium [27], ConvNeXtV2-Large [28] e DinoV2 [29] a partir do dataset de teste.
- Mostrar as estratégias de fusões intermediárias, ao combinar os resultados das camadas intermediárias das redes neurais ConvNeXtV2-Large [28] com DinoV2 [29], DinoV2 [29] com EfficientNetV2-Medium [27], ConvNeXtV2-Large [28] com EfficientNetV2-Medium [27] e por último a fusão das redes EfficientNetV2-Medium [27] com ConvNeXtV2-Large [28] com DinoV2 [29].
- Os resultados das estratégias de fusões serão avaliados pelas métricas de classificação acurácia, sensibilidade, especificidade, precisão, taxa de falsos positivos, F-escore, coeficiente de correlação de Matthews, Kappa e áreas sob as curvas da Característica de Operação do Receptor e da Precisão-Sensibilidade.


## 3. Estratégias de Fusões

### 3.1. Estratégia *Intermediate Fusion*

Na abordagem *Intermediate Fusion*, as imagens do *dataset* de treino passam pelas redes neurais para extração de características e, posteriormente, as características são concatenadas e processadas por uma outra rede neural, que gera os escores preditivos da classificação final [20-22], conforme a Figura 1.

![Figura 1: Representação da estratégia intermediate fusion, a usar a mesma imagem do gastrointestinal [19] como entrada para cada modelo de Deep Learning, com a finalidade de extrair características. Tais características são concatenadas e inseridas em uma nova rede neural.
](images/Figure_1.jpeg)

**Figura 1:** Representação da estratégia *Intermediate Fusion*, a usar uma imagem do *dataset* gastrointestinal [19] como entrada para cada modelo de *Deep Learning*, com a finalidade de extrair características. Tais características são concatenadas e inseridas em uma nova rede neural.

### 3.2. Estratégia *Late Fusion*

Diferentemente da estratégia *intermediate fusion*, a estratégia *late fusion* é a combinação dos resultados de predições com o *dataset* de teste das classificadores individuais já treinados com o *dataset* de treino [23-26].

![Figura 2: Representação da estratégia late fusion, a usar a usar a mesma imagem do gastrointestinal [19] como entrada para cada modelo de Deep Learning, que fornecem os escores preditos das classificações individuais. Tais escores são agrupados por alguma estratégia de fusão posterior.](images/Figure_2.jpeg)

**Figura 2:** Representação da estratégia *late fusion*, a usar uma mesma imagem do *dataset* gastrointestinal [19] como entrada para cada modelo de *Deep Learning*, que fornecem os escores preditos das classificações individuais. Tais escores são agrupados por alguma estratégia de fusão tardia.

Matematicamente, tais combinações dos escores podem ser definidas como o valor médio, máximo, mínimo e produto normalizado (ProNor) dos modelos de *deep learning*, conforme as equações (1), (2), (3) e (4) respectivamente [30-33]:

<img src="images/equacao1_media_branca.png" alt="Equação 1 - Média" width="270">

<img src="images/equacao2_maximo_branca.png" alt="Equação 2 - Máximo" width="260">

<img src="images/equacao3_minimo_branca.png" alt="Equação 3 - Mínimo" width="260">

<img src="images/equacao4_pronor_branca.png" alt="Equação 4 - ProNor" width="200">

---

Matrícula: 242.100.269

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Visão Computacional Master*

