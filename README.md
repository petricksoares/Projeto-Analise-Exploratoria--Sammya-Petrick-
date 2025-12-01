# Projeto Análise Exploratória

## Objetivo do Projeto🎯
O objetivo deste estudo é investigar os padrões que influenciam a experiência e a satisfação do cliente no e-commerce brasileiro, com foco em atrasos na entrega, níveis de satisfação, diferenças de preço e de frete, categorias de produtos problemáticas e variações no tempo de processamento e envio dos pedidos, visando fins de estudo e análise.

## 🔗 Base de Dados Utilizada
*Olist Brazilian E-Commerce Dataset*  
Disponível em: [Kaggle - Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

### Datasets Utilizados:
- olist_orders_dataset.csv
- olist_order_items_dataset.csv 
- olist_products_dataset.csv

## Descrição do Processo de Tratamento dos Dados
O processo de tratamento dos dados começou com o carregamento dos três datasets obrigatórios e sua junção pelas chaves order_id e product_id, resultando em um dataset final unificado com 112.650 registros. Em seguida, foi realizada uma análise exploratória, examinando a estrutura do dataset, os tipos de dados, valores ausentes e estatísticas descritivas das variáveis numéricas.

A limpeza dos dados incluiu o tratamento de valores ausentes por categoria, a correção de inconsistências como produtos com peso zero e a identificação de outliers, que foram tratados usando o método Z-score com capping em ±3 desvios padrão. As datas foram convertidas para o formato datetime, e os nomes de colunas foram padronizados e corrigidos, aplicando-se ainda técnicas de normalização MinMax e padronização Z-score para variáveis numéricas.

Para os dados categóricos, utilizou-se Label Encoding para a coluna product_category_name e One-Hot Encoding para order_status. Foram criadas novas features, incluindo tempo de atraso entre a entrega real e estimada, proporção entre frete e preço, densidade do produto e tempo de processamento, com o objetivo de enriquecer a análise logística e de eficiência. Por fim, realizou-se a seleção de atributos com base em correlações, variância e significância estatística, garantindo que apenas as variáveis mais relevantes fossem consideradas para análises futuras.

## Principais Desafios Encontrados!
### 1. Tratamento de Valores Ausentes
Foram identificados 1.603 registros sem categoria de produto, além de datas de entrega ausentes em pedidos não concluídos. Para contornar essas lacunas, as categorias foram preenchidas com "sem_categoria" e as datas, com a mediana temporal correspondente.

### 2. Gestão de Outliers
Alguns produtos apresentaram valores extremos de preço, chegando a R$ 6.735, e fretes desproporcionais em relação ao valor dos produtos. Para preservar a distribuição dos dados, aplicou-se um capping estatístico, mantendo a consistência das análises.

### 3. Feature Engineering
Foram criadas métricas temporais consistentes, incluindo o cálculo de densidade para produtos com dimensões irregulares, e normalizadas proporções de frete e preço para permitir comparações justas entre diferentes itens.

### 4. Dimensionalidade
A coluna de categoria de produtos possuía 72 categorias distintas. Buscando equilibrar riqueza de informação e complexidade, decidiu-se manter todas as categorias, permitindo uma análise setorial detalhada.

## 📈 Conclusões Finais
No estudo, foi analisado o desempenho do e-commerce brasileiro, com foco em atrasos de entrega, custos de frete e eficiência operacional. Observou-se que 6,8% dos pedidos tiveram atraso, sendo o tempo de processamento um fator determinante, e que algumas categorias de produtos apresentam maior incidência de problemas logísticos.

Em relação aos custos, identificou-se uma correlação moderada (0,329) entre preço e frete, com a proporção média de frete em relação ao preço de 25,4%. Algumas categorias apresentaram fretes desproporcionais, sugerindo oportunidades para ajustes que aumentem a competitividade.

O dataset possui boa qualidade geral, com poucos valores problemáticos após o tratamento, e as features criadas durante a análise enriquecem a compreensão sobre atrasos, custos e eficiência interna.

Esses insights apontam que a otimização logística pode reduzir atrasos, a revisão de fretes em categorias específicas pode melhorar a competitividade, e o monitoramento do tempo de processamento é essencial para garantir uma experiência satisfatória ao cliente.

## 🛠 Tecnologias Utilizadas
- *Python 3.x*
- *Pandas* - Manipulação de dados
- *NumPy* - Cálculos numéricos
- *Matplotlib/Seaborn* - Visualizações
- *Scikit-learn* - Pré-processamento (apenas)
- *Google Colab* - Ambiente de execução

## 👥 Integrantes da Dupla
- *Sammya* 
- *Petrick*
---

*Desenvolvido por Sammya e Petrick*  
Última atualização: Dezembro 2025
