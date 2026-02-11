# 🏥 Predição de Risco Clínico COVID-19: Priorizando Vidas com Machine Learning

## 📝 Visão Geral
Este projeto desenvolve um modelo preditivo baseado em inteligência artificial para auxiliar na triagem de pacientes com Síndrome Respiratória Aguda Grave (SRAG) em Recife. O objetivo principal é **identificar precocemente indivíduos com alto risco de óbito**, permitindo uma alocação de recursos hospitalares mais eficiente e segura.

O modelo final atingiu um **Recall de 96%**, garantindo que quase a totalidade dos casos críticos seja identificada para intervenção imediata.
   <div align="center">
  <img width="611" height="455" alt="Matriz de Confusão XGBoost" src="https://github.com/user-attachments/assets/2c4c062e-ee93-427d-9d75-a3fc2184e055" />
  <p><i>Figura 1: Matriz de Confusão do XGBoost evidenciando o Recall de 96% para a classe OBITO.</i></p>
</div>

---
## 🎯 Problema de Negócio e Impacto
Em cenários de crise sanitária, o tempo de resposta é crucial. O desafio consistia em processar dados reais de saúde pública que possuíam:

1. **Desacoplamento entre Triagem e Desfecho:** Conforme o [Protocolo de Manejo Clínico do Ministério da Saúde (SRAG 2025)](https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/guias-e-manuais/2025/guia-de-orientacoes-para-profissionais-de-saude-srag.pdf), a classificação inicial (Leve vs. Grave) baseia-se apenas nos sintomas de entrada. O modelo aqui desenvolvido busca preencher a lacuna preditiva, antecipando o desfecho final (Recuperação vs. Óbito), independentemente da classificação de gravidade inicial.
2. **Desbalanceamento Severo:** A classe de óbitos é significativamente menor que a de recuperados (minorante), exigindo técnicas específicas de ajuste de pesos e métricas de avaliação.
   
   <img width="590" height="490" alt="download" src="https://github.com/user-attachments/assets/cc718659-ac91-4b1a-9a00-3ebd49a30774" />

4. **Dados Desestruturados:** Sintomas registrados em listas de strings, demandando engenharia de variáveis para extração de sinais clínicos relevantes.
5. **Prioridade Ética:** No contexto hospitalar, o custo de um **Falso Negativo** (não identificar um potencial óbito) é humanamente superior ao custo de um Falso Positivo (monitoramento preventivo de um paciente que se recuperaria).
---

## 🛠️ Engenharia de Dados (Data Prep & Intelligence)
A performance do modelo foi impulsionada por um tratamento de dados rigoroso e estratégico, focado em extrair valor de bases de saúde pública:

* **Curadoria de Features:** Seleção criteriosa de colunas baseada na relevância para o desfecho clínico, taxa de preenchimento (missing values) e variância (eliminando colunas constantes que não agregavam poder preditivo).
* **Tratamento de Dados Faltantes:** A imputação não foi feita de forma automatizada; cada coluna foi avaliada individualmente conforme seu impacto semântico e a distribuição das faltas, preservando a integridade estatística.
* **Padronização e Alinhamento:** * Unificação de esquemas entre diferentes bases de dados (alinhamento de nomes de colunas).
    * Limpeza de valores categóricos (ex: harmonização de acentuação em 'INDÍGENA').
    * **Agrupamento Semântico:** Consolidação de termos correlatos (ex: agrupamento de 'Distúrbios Olfativos' e 'Gustativos' na nova feature `sintoma_perda_sentidos`) para reduzir ruído e aumentar a robustez do modelo.
* **Extração de Sintomas (Multi-label):** Processamento da coluna original de sintomas (que continha listas de strings) em variáveis binárias independentes (*dummies*), permitindo que o modelo interpretasse a presença de sinais de alerta como **Dispneia** e **Saturação de O2** de forma isolada.
* **Feature Engineering de Datas:** Cálculo do intervalo entre o início dos sintomas e a notificação. Aplicou-se técnicas de *clipping* e *binning* para mitigar *outliers* de preenchimento e capturar o impacto do atraso no atendimento como um fator de risco latente.
* **Arquitetura de Processamento:** Implementação via `ColumnTransformer` e `Pipeline` do Scikit-Learn, garantindo que o pré-processamento fosse isolado dentro da validação cruzada, evitando o **Data Leakage** (vazamento de dados)..

---
## 📊 Análise Exploratória de Dados (EDA)

Abaixo, os principais insights extraídos que fundamentaram as decisões de Engenharia de Dados e a seleção de variáveis para o modelo:

### 1. Perfil Etário e Letalidade
Há uma diferença clara entre os dois grupos: quanto maior a idade, maior a probabilidade de óbito. O gráfico abaixo mostra o deslocamento da curva de letalidade conforme o avanço das faixas etárias.

<div align="center">
  <img width="800" src="https://github.com/user-attachments/assets/11b7cf30-91ec-4c93-8a1e-6e366da5e821" />
</div>

### 2. Sintomatologia Crítica
De acordo com o manual do Ministério da Saúde, os sintomas que exigem maior atenção são **Dispneia**, **Desconforto Respiratório** e **Saturação de O2 ≤ 94%**. Os dados respaldam essa diretriz: a presença desses sinais clínicos aumenta drasticamente a probabilidade de óbito.

<div align="center">
  <img width="700" src="https://github.com/user-attachments/assets/751c1e27-d7f2-459d-9953-7caba6e93a0f" />
</div>

### 3. Fatores Demográficos (Sexo e Raça)
* **Sexo:** Não fornece uma distinção tão expressiva quanto a idade, mas os dados revelam que homens possuem uma chance ligeiramente superior de ir a óbito em comparação às mulheres.
<div align="center">
  <img width="600" src="https://github.com/user-attachments/assets/0d94cda2-7fc5-4858-8d66-64cc89a1e97d" />
</div>

* **Raça:** O manual do Ministério da Saúde destaca vulnerabilidades específicas para populações indígenas ou com dificuldade de acesso. Como as variações visuais no gráfico eram sutis, apliquei um teste estatístico para confirmar a relevância:
    * **Estatística Chi-Quadrado ($\chi^2$):** 1646.61
    * **p-value:** 0.0000e+00
    * **Graus de Liberdade:** 5  
> O resultado confirma que a variável raça possui impacto estatisticamente significativo no desfecho dos casos.

<div align="center">
  <img width="700" src="https://github.com/user-attachments/assets/a98e73dd-2bce-40a5-a343-c706ee52805e" />
</div>

### 4. Intervalo de Notificação (Sintoma até Registro)
Esta variável mostrou-se extremamente ruidosa e repleta de *outliers*. No entanto, é possível detectar comportamentos distintos entre as classes. Quando bem tratada via *clipping* e *binning*, essa feature auxilia o modelo na separação dos grupos.

<div align="center">
  <img width="800" src="https://github.com/user-attachments/assets/a9b277ff-c786-4efd-83c8-1178e0284865" />
</div>

---

---
## 📊 Experimentos e Performance

Foram realizados experimentos comparativos entre **Random Forest** e **XGBoost**, utilizando a biblioteca **Optuna** para o ajuste fino de hiperparâmetros (Otimização Bayesiana).

### Resultado Final (Conjunto de Teste - Dados Não Vistos)
Optou-se pelo modelo **XGBoost Otimizado para Recall**, priorizando a segurança clínica e a redução drástica de Falsos Negativos.

| Métrica | Performance |
| :--- | :--- |
| **Recall (Sensibilidade)** | **96%** |
| **Precisão (Óbito)** | **52%** |
| **F1-Score** | **67%** |
| **Acurácia Global** | **92%** |

> **Veredito:** O modelo mantém uma precisão de 52% mesmo com um recall altíssimo. Na prática, isso significa que em uma triagem, o modelo acerta 1 a cada 2 alertas de risco, enquanto deixa passar apenas 4% dos casos fatais.

### 🔍 Interpretabilidade e Maiores Preditores
O gráfico de importância de variáveis revela que o modelo prioriza os **sintomas críticos**, mas traz um *insight* valioso sobre a qualidade dos dados:

* **Sintomas no Topo:** Saturação, Dispneia e Aperto Torácico dominam a predição, validando a eficácia da nossa Engenharia de Dados.
* **O fator 'Raça Ignorado':** Surpreendentemente, a ausência de informação sobre a raça do paciente (`raca_ignorado`) apareceu como um preditor mais forte que a própria `idade`. 
    * **Hipótese Analítica:** Em cenários de crise, o preenchimento incompleto de fichas costuma ocorrer em hospitais sobrecarregados. Assim, o dado ignorado atua como um indicador indireto para a **pressão no sistema de saúde**, correlacionando-se com casos de maior gravidade onde o tempo para burocracia era escasso.
* **Idade e Tempo de Notificação:** Seguem como preditores fundamentais, confirmando o perfil biológico e logístico do risco.

<div align="center">
  <img width="900" alt="Feature Importance XGBoost" src="https://github.com/user-attachments/assets/3767015b-b179-46a0-8481-a0e56cca4f25" />
  <p><i>Feature Importance: Note a relevância de 'raca_ignorado', sugerindo correlação entre falta de dados e gravidade do cenário hospitalar.</i></p>
</div>

---

## ⚖️ Decisão Estratégica
A decisão de priorizar o **Recall** fundamenta-se em protocolos de saúde pública. Em uma pandemia, é preferível que o sistema de saúde monitore preventivamente um paciente que se recuperaria do que negligenciar um paciente em estado crítico por falha de detecção.

---
## ⚠️ Limitações do Estudo

Apesar dos excelentes resultados em sensibilidade (Recall), o projeto possui limitações que devem ser consideradas para implementações futuras:

1.  **Ausência de Comorbidades:** O dataset original apresentava lacunas críticas em dados sobre doenças prévias (diabetes, hipertensão, etc.). A inclusão desses dados estruturados poderia elevar significativamente a **Precisão** do modelo sem sacrificar o Recall.
2.  **Validação Clínica:** Este modelo atua como uma ferramenta de suporte à decisão. Sua aplicação prática exige validação por especialistas do domínio (médicos e epidemiologistas) para garantir a plausibilidade biológica das correlações encontradas.
3.  **Dados de Texto Livre:** A extração de sintomas dependeu da qualidade do preenchimento dos profissionais na ponta. Erros de digitação ou omissões nos registros de "sintomas" podem impactar o desempenho em tempo real.
4.  **Janela Temporal:** O modelo foi treinado com dados de um período específico da pandemia. Mudanças nas variantes do vírus ou nos protocolos de vacinação podem exigir o retreinamento do modelo para manter a eficácia.
   
---

## 👨‍💻 Como Reproduzir
1. Clone o repositório:
   ```bash
      git clone https://github.com/mateusnotgado/COVID.git
