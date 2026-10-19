# Industrial Intelligence

Dashboard analítico industrial desenvolvido em Python com Streamlit, Pandas, Plotly, SQLite e Scikit-learn.

## Execução (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

O banco SQLite é gerado automaticamente na primeira execução. Para recriar os dados, execute `python generate_data.py` no ambiente virtual e reinicie o dashboard.

## Funcionalidades

- Visão geral com indicadores e gráficos de falhas
- Monitoramento de temperatura, vibração, pressão e carga
- Análise de falhas e horas de parada por ativo
- Classificação experimental de risco com Random Forest
- Exportação de dados filtrados para CSV
- Filtros globais por unidade, equipamento e período

## Dados e limitações

Dados 100% sintéticos, com semente fixa, gerados para fins educacionais. Os eventos de falha são simulados por uma função probabilística. O modelo de Machine Learning reproduz padrões artificiais da simulação e não é validado para uso industrial. As leituras representam observações simuladas e não uma série temporal contínua por equipamento; horas de parada são eventos simulados, não uma medição de disponibilidade operacional real. Os resultados de classificação não devem ser interpretados como previsões de falhas futuras com horizonte definido.

## Estrutura

- `app.py`: dashboard e modelo de classificação
- `generate_data.py`: gerador de dados sintéticos e banco SQLite
- `requirements.txt`: dependências
- `industrial.db`: banco gerado ao executar


### Correção de tema (v1.1)
As cores usam as variáveis nativas do Streamlit e os gráficos Plotly utilizam `theme="streamlit"`, permitindo alternar entre Light e Dark no menu do aplicativo.
