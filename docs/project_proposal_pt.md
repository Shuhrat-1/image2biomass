# Image2Biomass: Previsão de Biomassa de Pastagem a partir de Fotografias de Quadrats

**Categoria do projeto:** Regressão de imagem (com baseline tabular de metadados)

**Membros da equipa:** [Shuhrat Maksumov / ID: 28548], [Xavier Loreto / ID: 28648]

---

## Plano do projeto

**Enunciado do problema.** Pretendemos prever a biomassa de pastagem a partir
de fotografias de quadrats de campo de 70x30 cm, utilizando o conjunto de dados
CSIRO Image2Biomass. Cada imagem tem cinco variáveis-alvo de regressão em gramas:
matéria seca verde, morta, de trevo, GDM e total. A estimativa rigorosa da
biomassa ajuda os agricultores a decidir quando e como apascentar o gado, o que
torna este um problema agrícola de relevância prática. Para além da previsão, a
nossa questão científica central é uma comparação de modalidades: quanto é que a
imagem acrescenta relativamente a duas medições de campo de baixo custo (NDVI e
altura da pastagem)?

**Desafios.** O conjunto de dados é pequeno (357 imagens etiquetadas), o que
torna as redes profundas propensas a sobreajuste. As cinco variáveis-alvo
apresentam forte assimetria à direita (assimetria 1,4-2,8) com muitos zeros
(trevo ~38%), pelo que treinamos com uma transformação log1p. As observações
estão correlacionadas por local e por espécie, pelo que uma divisão aleatória
ingénua provoca fuga de informação; recorremos a validação cruzada por grupos.
Por fim, o conjunto de teste oculto do Kaggle não inclui metadados, pelo que um
estudo de modalidades justo deve comparar os modelos nos dados etiquetados
localmente, e não na tabela de classificação.

**Conjunto de dados.** CSIRO Image2Biomass ([Kaggle](https://www.kaggle.com/competitions/csiro-biomass/overview)): 357 fotografias de quadrats com biomassa de referência associada, mais metadados 
(estado, espécie, NDVI pré-pastoreio, altura média, data). Dividimos o conjunto etiquetado com
StratifiedGroupKFold por imagem, estratificado pela biomassa total agrupada em
intervalos, de modo a que a CNN e o baseline tabular sejam avaliados exatamente
nas mesmas partições.

**Método.** Construímos uma comparação em quatro níveis alinhada com a unidade
curricular: (1) um baseline ingénuo pela mediana; (2) um baseline tabular com
RandomForest / gradient boosting sobre os metadados (sessões 4, 7); (3) uma CNN
treinada de raiz (sessão 10); e (4) aprendizagem por transferência através do
ajuste fino (fine-tuning) de uma ResNet18 pré-treinada, comparada com a extração
de características com a rede congelada (sessão 12). Uma aplicação Gradio
publicada no Hugging Face (sessão 11) demonstra o modelo baseado apenas na
imagem no cenário realista do agricultor. Todas as variáveis-alvo são modeladas
no espaço log1p; as previsões são invertidas e limitadas a valores não negativos.

**Avaliação.** Reportamos um RMSE ponderado ao estilo da competição (pesos das
variáveis-alvo 0,1/0,1/0,1/0,2/0,5), bem como RMSE, MAE e R2 por variável-alvo na
escala original, agregados ao longo de 5 partições (média +/- desvio-padrão).
Comparamos todos os modelos nas mesmas partições e analisamos onde cada
modalidade tem sucesso ou falha (por exemplo, a matéria morta e o trevo são os
mais difíceis a partir da imagem). O principal resultado é uma resposta
quantificada à questão de saber se a imagem acrescenta valor preditivo para além
das medições de campo de baixo custo.
