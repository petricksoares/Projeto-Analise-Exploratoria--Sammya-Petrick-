# Conectando o colab com o drive
from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/trabalhoHeloísa/trabalho'

# importando biblioteca Pandas
import pandas as pd

# Carregando os três datasets
order_items = pd.read_csv(path+'/olist_order_items_dataset.csv')
orders = pd.read_csv(path+'/olist_orders_dataset.csv')
products = pd.read_csv(path+'/olist_products_dataset.csv')

# ESTRUTURA DOS DATASETS:
# order_items: contém 'order_id' e 'product_id' - é a tabela que liga pedidos aos produtos
# orders: contém 'order_id' (chave primária) e informações sobre o pedido (data, status, etc.)
# products: contém 'product_id' (chave primária) e informações sobre o produto (categoria, peso, etc.)

# ESTRATÉGIA DE JUNÇÃO:
# Primeiro juntamos 'order_items' com 'orders' usando a coluna 'order_id'
# Isso nos dá informações dos pedidos + itens dos pedidos
# Depois juntamos o resultado com 'products' usando a coluna 'product_id'
# Isso adiciona as informações completas dos produtos

# Unindo os datasets em um só DataFrame
df_completo = (order_items.merge(orders, on='order_id', how='left').merge(products, on='product_id', how='left'))    
# Adiciona infos detalhadas dos produtos
# Junta itens do pedido com infos do pedido

# Observando dataset
df_completo.info()
# Visualizar as primeiras 5 linhas do dataset
df_completo.head()
df_completo.describe()
# Exploração dos Dados
# Dataset Olist unificado
# Linhas: 112.650
# Colunas: 22

# Tipos de dados
# 12 colunas 'object' → contêm IDs, categorias, datas.
# 10 colunas numéricas → medidas, preços, pesos.
# Colunas datetime ainda estão como 'object', então precisaremos converter:
# shipping_limit_date
# order_purchase_timestamp
# order_approved_at
# order_delivered_carrier_date
# order_delivered_customer_date
# order_estimated_delivery_date

# 2. Valores Ausentes
# order_approved_at                  | 15
# order_delivered_carrier_date       | 1.194
# order_delivered_customer_date      | 2.454
# product_category_name              | 1.603
# product_name_lenght                | 1.603
# product_description_lenght         | 1.603
# product_photos_qty                 | 1.603
# product_weight_g                   | 18
# product_length_cm                  | 18
# product_height_cm                  | 18
# product_width_cm                   | 18

# Convertendo as colunas que precisam estar como datetime64[ns]
# Essas colunas estão como object e o pandas trata como strings, então elas precisam ser convertidas
# Conversão da coluna 'shipping_limit_date'
df_completo['shipping_limit_date'] = df_completo['shipping_limit_date'].astype('datetime64[ns]')

# Conversão da coluna 'order_purchase_timestamp'
df_completo['order_purchase_timestamp'] = df_completo['order_purchase_timestamp'].astype('datetime64[ns]')

# Conversão da coluna 'order_approved_at'
df_completo['order_approved_at'] = df_completo['order_approved_at'].astype('datetime64[ns]')

# Conversão da coluna 'order_delivered_carrier_date'
df_completo['order_delivered_carrier_date'] = df_completo['order_delivered_carrier_date'].astype('datetime64[ns]')

# Conversão da coluna 'order_delivered_customer_date'
df_completo['order_delivered_customer_date'] = df_completo['order_delivered_customer_date'].astype('datetime64[ns]')

# Conversão da coluna 'order_estimated_delivery_date'
df_completo['order_estimated_delivery_date'] = df_completo['order_estimated_delivery_date'].astype('datetime64[ns]')
df_completo.info()
# verificando a conversão

# Duas colunas tão com os nomes errados então nós corrigimos 
df_completo = df_completo.rename(columns={
'product_name_lenght': 'product_name_length',
'product_description_lenght': 'product_description_length'})
# Vendo se os nomes das colunas foram corrigidos 
df_completo.info()

# LIMPEZA DE DADOS
# Verificando quantas linhas duplicadas existem
duplicadas = df_completo.duplicated().sum()
duplicadas
# Sem nenhuma duplicadas
# linhas duplicadas
df_completo[df_completo.duplicated()]
# Sem linhas duplicadas

# Alguma colunas não podem ter valores nulos ou 0, como a coluna de preço e peso entre outras. Então foi feita aqui a soma de valores nulos/0 dessas colunas
print("nulos ou zerados na coluna preco =", df_completo["price"].isnull().sum())
print("nulos ou zerados na coluna frete =", df_completo["freight_value"].isnull().sum())
print("nulos ou zerados na coluna peso =", df_completo["product_weight_g"].isnull().sum())
print("nulos ou zerados na coluna altura =", df_completo["product_height_cm"].isnull().sum())
print("nulos ou zerados na coluna largura =", df_completo["product_width_cm"].isnull().sum())
print("nulos ou zerados na coluna comprimento =", df_completo["product_length_cm"].isnull().sum())

# Essa coluna não pode ser preenchida com o valor da media da coluna
# Ja que alguns produtos tem pesos muito diferentes
df_completo[df_completo["product_weight_g"] == 0].head(20)
# A coluna 'product_length_cm' que é da categoria 'cama_mesa_banho' tem uma inconsistência que são valores nulos
# Para resolver isso a gnt verificou a media da coluna e substituiu os valores nulos

# Descobrindo a media da coluna
mediana_peso_categoria = df_completo[df_completo["product_category_name"] == "cama_mesa_banho"]["product_weight_g"].median()
mediana_peso_categoria

# Substituindo peso = 0 pela mediana da categoria
df_completo.loc[df_completo["product_weight_g"] == 0, "product_weight_g"] = 1275.0

# Conferindo se ainda existem zeros
(df_completo["product_weight_g"] == 0).sum()
df_completo.info()

# Tratamento de valores ausentes
# Categóricas: preencher com "sem_categoria"
df_completo['product_category_name'] = df_completo['product_category_name'].fillna('sem_categoria')

# Tratamento de valores ausentes
# Preencher nulos com a media da coluna
# order_approved_at → preencher com a data de compra (faz sentido)
df_completo['order_approved_at'] = df_completo['order_approved_at'].fillna(df_completo['order_purchase_timestamp'])

# order_delivered_carrier_date → preencher com a média real da coluna (não depende da categoria)
media_carrier = df_completo['order_delivered_carrier_date'].median()
df_completo['order_delivered_carrier_date'] = df_completo['order_delivered_carrier_date'].fillna(
    media_carrier
)

# order_delivered_customer_date → preencher com a média real da coluna
media_customer = df_completo['order_delivered_customer_date'].median()
df_completo['order_delivered_customer_date'] = df_completo['order_delivered_customer_date'].fillna(
    media_customer
)

# Primeiro, vamos ver o que fazer com a data de aprovação do pedido.
# Muitos pedidos nulos nessa coluna são porque o pagamento não foi aprovado
# ou a informação se perdeu.
# Faz mais sentido preencher a data de aprovação com a própria data da compra.
# Assim, assumimos que a aprovação foi quase instantânea.

# A coluna de referência (o que vamos usar para preencher) é 'order_purchase_timestamp'.
coluna_data_compra = df_completo['order_purchase_timestamp']

# Preenchemos a coluna de aprovação com a coluna de compra, só onde for nulo.
df_completo['order_approved_at'] = df_completo['order_approved_at'].fillna(coluna_data_compra)

# Agora, a data que o produto foi entregue para a transportadora.
# Como é difícil chutar essa data, vamos usar um valor 'central' para não bagunçar
# a análise dos tempos de entrega.
# Usar a MEDIANA é mais seguro que a média, porque a mediana não é afetada
# por entregas muito rápidas ou muito demoradas (os 'outliers').

# 1. Calculamos o valor central (mediana) da coluna:
valor_mediano_entrega_transportadora = df_completo['order_delivered_carrier_date'].median()

# 2. Preenchemos os valores nulos com esse valor central:
df_completo['order_delivered_carrier_date'] = df_completo['order_delivered_carrier_date'].fillna(valor_mediano_entrega_transportadora)

# Por fim, vamos preencher a data que o produto chegou na casa do cliente.
# A lógica é a mesma da transportadora: vamos usar o valor MEDIANO
# de todas as entregas já feitas.

# 1. Calculamos o valor central (mediana) da coluna de entrega final:
valor_mediano_entrega_cliente = df_completo['order_delivered_customer_date'].median()

# 2. Preenchemos os valores nulos com esse valor central que achamos:
df_completo['order_delivered_customer_date'] = df_completo['order_delivered_customer_date'].fillna(
    valor_mediano_entrega_cliente
)
df_completo.info()

# Percebi que algumas colunas ainda estão com valores ausentes:
# product_name_length → 1.603 nulos
# product_description_length → 1.603 nulos
# product_photos_qty → 1.603 nulos
# As outras numéricas (product_weight_g, product_length_cm, product_height_cm, product_width_cm) já foram completamente preenchidas.
# Isso aconteceu pq a mediana por categoria não pode ser calculada para algumas categorias que só tinham valores ausentes ou poucos registros.

# Preencher qualquer NaN restante com mediana global
colunas_restantes = ['product_name_length', 'product_description_length', 'product_photos_qty']

for coluna in colunas_restantes:
 mediana_global = df_completo[coluna].median()
 df_completo[coluna] = df_completo[coluna].fillna(mediana_global)
df_completo.info()

# Aqui eu pego só as colunas numéricas que fazem sentido pra procurar outliers.
# Depois calculo o Z-score de cada uma, que basicamente diz o quão longe um valor está da média.
# Se o Z-score passar de 3 (positivamente ou negativamente), eu marco como outlier.
# No final, mostro quantos outliers cada coluna tem.
import numpy as np
colunas_numericas = ['price', 'freight_value','product_name_length', 'product_description_length', 'product_photos_qty',
'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']

# DataFrame só com z-scores
z_scores = (df_completo[colunas_numericas] - df_completo[colunas_numericas].mean()) / df_completo[colunas_numericas].std()

# Identificar outliers
outliers_zscore = (np.abs(z_scores) > 3)

# Contagem por coluna
print("Quantidade de outliers por Z-score:")
print(outliers_zscore.sum())

# Colunas que vamos tratar com capping
colunas_numericas = ['price', 'freight_value','product_name_length', 'product_description_length', 'product_photos_qty',
'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']

# Percorre cada coluna e aplica capping usando Z-score 3
for col in colunas_numericas:
   # Calcula média e desvio padrão da coluna
   media = df_completo[col].mean()
   desvio = df_completo[col].std()

   # Define os limites inferior e superior (3 desvios da média)
   limite_inferior = media - 3*desvio
   limite_superior = media + 3*desvio

   # Substitui valores abaixo do limite inferior pelo limite inferior
   # e valores acima do limite superior pelo limite superior
   df_completo[col] = np.where(df_completo[col] > limite_superior, limite_superior, df_completo[col]) 
   df_completo[col] = np.where(df_completo[col] < limite_inferior, limite_inferior, df_completo[col])

import matplotlib.pyplot as plt

for col in colunas_numericas:
  plt.figure(figsize=(6,3))
  plt.boxplot(df_completo[col])
  plt.title(f'Boxplot - {col}')
  plt.show()

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1 Transformar categorias de produto em números
# Cada categoria de produto vai virar um número. Bem simples de entender.
# Por exemplo: "cama_mesa_banho" → 0, "perfumaria" → 1, etc.
le = LabelEncoder()
df_completo['categoria_produto'] = le.fit_transform(df_completo['product_category_name'])
print("Label Encoding feito para 'product_category_name'. Aqui estão alguns exemplos:")
print(df_completo[['product_category_name', 'categoria_produto']].head())
print()

# Transformar status do pedido em colunas separadas com 0 ou 1
# Cada status vira uma coluna, mostrando se o pedido tem ou não aquele status.
# Exemplo: status_delivered = 1 se entregue, 0 se não.
# Verifica se a coluna 'order_status' existe antes de aplicar get_dummies
if 'order_status' in df_completo.columns:
    df_completo = pd.get_dummies(df_completo, columns=['order_status'], prefix='status')
    print("One-Hot Encoding feito para 'order_status'. Algumas das colunas criadas:")
    print([col for col in df_completo.columns if col.startswith('status_')][:5], "...")
else:
    print("A coluna 'order_status' já foi processada ou não existe.")

#  TRATAMENTO DOS DADOS CATEGÓRICOS E TEXTOS

print("Colunas categóricas:")
print(df_completo.select_dtypes(include=['object']).columns.tolist())

print("\nValores únicos em product_category_name:")
print(df_completo['product_category_name'].value_counts().head(10))

# CODIFICAÇÃO DE DADOS CATEGÓRICOS

from sklearn.preprocessing import LabelEncoder

# Codificação Label Encoding para product_category_name
le_categoria = LabelEncoder()
df_completo['product_category_encoded'] = le_categoria.fit_transform(
    df_completo['product_category_name'].fillna('sem_categoria')
)

# A coluna 'order_status' já foi processada por One-Hot Encoding em uma célula anterior (z9cv14cn3WIg).
# A remoção do código abaixo evita o KeyError.
# order_status_dummies = pd.get_dummies(df_completo['order_status'], prefix='status')
# df_completo = pd.concat([df_completo, order_status_dummies], axis=1)

print("Codificação concluída:")
print(f"Categories encoded: {len(le_categoria.classes_)}")
# print(f"Order status dummies: {order_status_dummies.columns.tolist()}") # Comentei para evitar erro, pois order_status_dummies não existe mais aqui.

#  NORMALIZAÇÃO E PADRONIZAÇÃO

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Colunas para normalização
colunas_numericas = ['price', 'freight_value', 'product_weight_g','product_length_cm', 'product_height_cm', 'product_width_cm']

# Padronização Z-score
scaler_z = StandardScaler()
df_completo[[f'{col}_zscore' for col in colunas_numericas]] = scaler_z.fit_transform(
    df_completo[colunas_numericas].fillna(df_completo[colunas_numericas].median())
)

# Normalização MinMax
scaler_minmax = MinMaxScaler()
df_completo[[f'{col}_minmax' for col in colunas_numericas]] = scaler_minmax.fit_transform(
    df_completo[colunas_numericas].fillna(df_completo[colunas_numericas].median())
)

print("Normalização e padronização concluídas")
print(df_completo[[f'{colunas_numericas[0]}_zscore', f'{colunas_numericas[0]}_minmax']].describe())

# SELEÇÃO DE ATRIBUTOS

# Análise de Correlação
import seaborn as sns
import matplotlib.pyplot as plt

colunas_corr = ['price', 'freight_value', 'product_weight_g', 'product_length_cm','product_height_cm', 'product_width_cm', 'product_photos_qty']

plt.figure(figsize=(10, 8))
correlation_matrix = df_completo[colunas_corr].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação - Atributos Numéricos')
plt.tight_layout()
plt.show()

# Identificação de Baixa Variância
variancias = df_completo[colunas_numericas].var()
print("\nVariância dos atributos numéricos:")
print(variancias.sort_values())

# Filtros Simples - Remover colunas com variância muito baixa
limite_variancia = 0.1
colunas_baixa_variancia = variancias[variancias < limite_variancia].index.tolist()
print(f"\nColunas com baixa variância (<{limite_variancia}): {colunas_baixa_variancia}")

# Manter apenas colunas com variância adequada
colunas_selecionadas = [col for col in colunas_numericas if col not in colunas_baixa_variancia]
print(f"Colunas selecionadas: {colunas_selecionadas}")

# CRIAÇÃO DE NOVOS ATRIBUTOS (FEATURE ENGINEERING)

# Técnica 1: Feature de Tempo - Atraso na Entrega
df_completo['atraso_entrega_horas'] = (
    df_completo['order_delivered_customer_date'] - df_completo['order_estimated_delivery_date']
).dt.total_seconds() / 3600
df_completo['teve_atraso'] = (df_completo['atraso_entrega_horas'] > 0).astype(int)

# Técnica 2: Feature de Proporção - Frete/Preço
df_completo['proporcao_frete_preco'] = df_completo['freight_value'] / df_completo['price']
df_completo['proporcao_frete_preco'] = df_completo['proporcao_frete_preco'].replace([np.inf, -np.inf], np.nan)

# Técnica 3: Feature de Densidade do Produto
df_completo['volume_cm3'] = (df_completo['product_length_cm'] *
                            df_completo['product_height_cm'] *
                            df_completo['product_width_cm'])
df_completo['densidade_g_cm3'] = df_completo['product_weight_g'] / df_completo['volume_cm3']
df_completo['densidade_g_cm3'] = df_completo['densidade_g_cm3'].replace([np.inf, -np.inf], np.nan)

# Técnica 4: Feature de Tempo de Processamento
df_completo['tempo_processamento_horas'] = (
    df_completo['order_approved_at'] - df_completo['order_purchase_timestamp']
).dt.total_seconds() / 3600

print("Feature Engineering concluído:")
print(f"- Atraso entrega: {df_completo['teve_atraso'].mean():.2%} dos pedidos com atraso")
print(f"- Proporção frete/preço média: {df_completo['proporcao_frete_preco'].median():.3f}")
print(f"- Densidade média: {df_completo['densidade_g_cm3'].median():.3f} g/cm³")

# PIPELINE COMPLETO DE PRÉ-PROCESSAMENTO

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Definir colunas para diferentes tratamentos
numeric_features = ['price', 'freight_value', 'product_weight_g', 'product_length_cm']
categorical_features = ['product_category_name']

# Criar transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("Pipeline de pré-processamento criado:")
print(preprocessor)

# VISUALIZAÇÕES E GRÁFICOS EXPLICATIVOS

# Visualização 1: Distribuição de Atrasos
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
df_completo['teve_atraso'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Distribuição de Pedidos com Atraso')

# Visualização 2: Atraso por Categoria
plt.subplot(2, 3, 2)
top_categories_atraso = df_completo.groupby('product_category_name')['teve_atraso'].mean().sort_values(ascending=False).head(10)
top_categories_atraso.plot(kind='barh', color='skyblue')
plt.title('Top 10 Categorias com Maior % de Atraso')
plt.xlabel('% de Atrasos')

# Visualização 3: Preço vs Frete
plt.subplot(2, 3, 3)
plt.scatter(df_completo['price'], df_completo['freight_value'], alpha=0.5, color='purple')
plt.xlabel('Preço do Produto (R$)')
plt.ylabel('Valor do Frete (R$)')
plt.title('Relação: Preço vs Frete')

# Visualização 4: Distribuição do Tempo de Entrega
plt.subplot(2, 3, 4)
df_completo['tempo_entrega_dias'] = df_completo['atraso_entrega_horas'] / 24
df_completo['tempo_entrega_dias'].hist(bins=50, color='orange', alpha=0.7)
plt.xlabel('Atraso (dias)')
plt.ylabel('Frequência')
plt.title('Distribuição do Atraso na Entrega')

# Visualização 5: Proporção Frete/Preço por Categoria
plt.subplot(2, 3, 5)
top_categories_frete = df_completo.groupby('product_category_name')['proporcao_frete_preco'].median().sort_values(ascending=False).head(10)
top_categories_frete.plot(kind='barh', color='lightgreen')
plt.title('Top 10 Categorias com Maior Frete/Preço')
plt.xlabel('Proporção Frete/Preço')

# Visualização 6: Densidade vs Peso
plt.subplot(2, 3, 6)
plt.scatter(df_completo['product_weight_g'], df_completo['densidade_g_cm3'], alpha=0.5, color='red')
plt.xlabel('Peso (g)')
plt.ylabel('Densidade (g/cm³)')
plt.title('Relação: Peso vs Densidade')

plt.tight_layout()
plt.show()

# RESPOSTA ÀS PERGUNTAS NORTEADORAS (4 de 6)=
print("RESPOSTAS ÀS PERGUNTAS NORTEADORAS")
# Pergunta 1: Quais características mais se relacionam com atrasos de entrega?
print("\n1. CARACTERÍSTICAS RELACIONADAS COM ATRASOS DE ENTREGA:")
correlacao_atraso = df_completo.corr(numeric_only=True)['atraso_entrega_horas'].sort_values(ascending=False)
print("Correlação com atraso na entrega:")
for feature, corr in correlacao_atraso.head(6).items():
    if feature != 'atraso_entrega_horas' and abs(corr) > 0.05:
        print(f"  {feature}: {corr:.3f}")

# Pergunta 2: Categorias com maior frequência de problemas
print("\n2. CATEGORIAS COM MAIOR FREQUÊNCIA DE PROBLEMAS:")
categorias_problemas = df_completo.groupby('product_category_name').agg({
    'teve_atraso': 'mean',
    'proporcao_frete_preco': 'median',
    'price': 'median'
}).round(3)

print("Top 5 categorias com maior % de atraso:")
print(categorias_problemas.nlargest(5, 'teve_atraso')[['teve_atraso', 'proporcao_frete_preco']])

# Pergunta 3: Tratamento de outliers
print("\n3. TRATAMENTO DE OUTLIERS:")
print("Foram identificados outliers usando Z-score > 3:")
print("Preço: 1.966 outliers tratados")
print("Frete: 2.041 outliers tratados")
print("Peso: 2.955 outliers tratados")
print("Método: Capping com limites de \u00b13 desvios padrão da média")

# Pergunta 4: Atributos com maior correlação
print("\n4. ATRIBUTOS COM MAIOR CORRELAÇÃO:")
print("Correlações mais significativas:")
corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
for pair, corr in corr_pairs.head(10).items():
    if pair[0] != pair[1] and abs(corr) > 0.3:
        print(f"  {pair[0]} vs {pair[1]}: {corr:.3f}")
