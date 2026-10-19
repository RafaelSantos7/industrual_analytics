from pathlib import Path
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from generate_data import generate, DB

st.set_page_config(page_title='Industrial Intelligence',page_icon=':material/precision_manufacturing:',layout='wide',initial_sidebar_state='expanded')
CSS='''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,.stApp,[data-testid="stSidebar"] {font-family:Inter,sans-serif;}
/* Native Streamlit theme tokens work in both Light and Dark. */
.stApp {background:var(--background-color);color:var(--text-color);}
[data-testid="stSidebar"] {background:var(--secondary-background-color);border-right:1px solid color-mix(in srgb,var(--text-color) 13%,transparent);}
[data-testid="stSidebar"] * {color:var(--text-color);}
[data-testid="stSidebar"] [data-baseweb="tag"] {background:color-mix(in srgb,var(--primary-color) 17%,var(--secondary-background-color));}
[data-testid="stSidebar"] [data-baseweb="tag"] span {color:var(--text-color);}
.block-container {padding-top:1.5rem;max-width:1550px;padding-left:2rem;padding-right:2rem;}
[data-testid="stMetric"] {background:var(--secondary-background-color);border:1px solid color-mix(in srgb,var(--text-color) 13%,transparent);border-radius:13px;padding:16px;min-width:0;}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color:var(--text-color);}
[data-testid="stMetricValue"] {font-size:clamp(1.3rem,2vw,1.8rem);font-weight:700;overflow-wrap:anywhere;}
h1,h2,h3 {color:var(--text-color)!important;letter-spacing:-.025em;}
h1 {font-size:2rem!important;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:color-mix(in srgb,var(--text-color) 13%,transparent)!important;border-radius:14px!important;background:var(--secondary-background-color);}
.stButton>button {border-radius:9px;}
@media(max-width:1100px){.block-container{padding-left:1rem;padding-right:1rem;}}
</style>'''
st.markdown(CSS,unsafe_allow_html=True)
if not DB.exists(): generate()
@st.cache_data
def load():
    with sqlite3.connect(DB) as conn:
        return pd.read_sql_query('SELECT s.*, e.unit, e.type, e.age_years FROM sensor_readings s JOIN equipment e ON e.equipment_id=s.equipment_id',conn,parse_dates=['date'])
raw=load()

# Icons use inline SVG from Lucide's public icon geometry, not emoji.
ICONS={'overview':'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
'monitor':'<path d="M3 12h4l3-8 4 16 3-8h4"/>','failure':'<path d="M12 9v4m0 4h.01M10.3 3.9 2 18a2 2 0 0 0 1.7 3h16.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
'predict':'<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 15l4-4 3 2 4-5"/>','data':'<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 4 18 4 18 0V5M3 12c0 4 18 4 18 0"/>'}
def ico(key):return f'<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[key]}</svg>'
with st.sidebar:
    st.markdown('### INDUSTRIAL INTELLIGENCE')
    st.caption('ANALYTICS PLATFORM · V1.0')
    pages=['Visão geral','Monitoramento','Análise de falhas','Manutenção preditiva','Exploração de dados']
    labels=['overview','monitor','failure','predict','data']
    page=st.radio('Navegação',pages,label_visibility='collapsed')
    st.divider()
    st.markdown('**Filtros globais**')
    units=st.multiselect('Unidades',sorted(raw.unit.unique()),default=sorted(raw.unit.unique()))
    types=st.multiselect('Equipamentos',sorted(raw.type.unique()),default=sorted(raw.type.unique()))
    min_date,max_date=raw.date.min().date(),raw.date.max().date()
    selected=st.date_input('Período',(min_date,max_date),min_value=min_date,max_value=max_date)
    st.divider()
    st.caption('Dados sintéticos · Uso educacional')
if not isinstance(selected,(list,tuple)) or len(selected)!=2:
    st.warning('Selecione um intervalo de datas completo.');st.stop()
df=raw[(raw.unit.isin(units))&(raw.type.isin(types))&(raw.date.dt.date>=selected[0])&(raw.date.dt.date<=selected[1])].copy()
if df.empty: st.info('Nenhum registro corresponde aos filtros selecionados.');st.stop()
COLORS=['#5865e8','#20a8a0','#f59e55','#e45c6a','#8b6bdc']
base=dict(template='plotly',color_discrete_sequence=COLORS)
def show(fig,height=310):
    fig.update_layout(height=height,margin=dict(l=8,r=8,t=26,b=5),font=dict(family='Inter',size=12),legend=dict(orientation='h',y=-.25),hovermode='closest')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(zeroline=False)
    st.plotly_chart(fig,use_container_width=True,theme='streamlit',config={'displayModeBar':False})
def panel(title,fig,height=310):
    with st.container(border=True):
        st.markdown(f'**{title}**');show(fig,height)

def headline(title,sub):
    st.title(title);st.caption(sub)

def kpis():
    count=df.equipment_id.nunique();fails=int(df.failure.sum());rate=100*df.failure.mean();down=df.downtime_hours.sum();
    cols=st.columns(5)
    for col,(label,value) in zip(cols,[('Equipamentos',f'{count}'),('Leituras',f'{len(df):,}'.replace(',','.')),('Eventos de falha',f'{fails:,}'.replace(',','.')),('Taxa de falha',f'{rate:.2f}%'),('Paradas registradas',f'{down:,.0f} h'.replace(',','.'))]):
        col.metric(label,value)

if page=='Visão geral':
    headline('Visão geral','Indicadores operacionais e desempenho dos ativos industriais')
    kpis();st.write('')
    a,b=st.columns([1.55,1])
    monthly=df.assign(month=df.date.dt.to_period('M').astype(str)).groupby('month',as_index=False).agg(falhas=('failure','sum'),leituras=('reading_id','count'))
    with a: panel('Evolução mensal das falhas',px.line(monthly,x='month',y='falhas',markers=True,labels={'month':'Mês','falhas':'Falhas'},**base))
    with b:
        u=df.groupby('unit',as_index=False).failure.sum().sort_values('failure')
        panel('Falhas por unidade',px.bar(u,x='failure',y='unit',orientation='h',labels={'failure':'Falhas','unit':''},**base))
    a,b=st.columns(2)
    with a:
        bytype=df.groupby('type',as_index=False).agg(taxa=('failure','mean'))
        bytype.taxa*=100
        panel('Taxa de falha por tipo (%)',px.bar(bytype,x='type',y='taxa',labels={'type':'Equipamento','taxa':'Taxa (%)'},**base))
    with b:
        hr=df.groupby(df.date.dt.dayofweek).agg(falhas=('failure','sum')).reset_index().rename(columns={'date':'weekday'})
        hr['dia']=hr.iloc[:,0].map(dict(enumerate(['Seg','Ter','Qua','Qui','Sex','Sáb','Dom'])))
        panel('Falhas por dia da semana',px.bar(hr,x='dia',y='falhas',labels={'dia':'Dia','falhas':'Falhas'},**base))
elif page=='Monitoramento':
    headline('Monitoramento de sensores','Distribuição e evolução das variáveis operacionais')
    cols=st.columns(4)
    for c,(label,field,unit) in zip(cols,[('Temperatura média','temperature_c','°C'),('Vibração média','vibration_mm_s','mm/s'),('Pressão média','pressure_bar','bar'),('Carga média','load_pct','%')]):c.metric(label,f'{df[field].mean():.1f} {unit}')
    a,b=st.columns(2)
    with a:panel('Temperatura por tipo de equipamento',px.box(df,x='type',y='temperature_c',color='type',labels={'type':'Equipamento','temperature_c':'Temperatura (°C)'},**base))
    with b:panel('Vibração por tipo de equipamento',px.box(df,x='type',y='vibration_mm_s',color='type',labels={'type':'Equipamento','vibration_mm_s':'Vibração (mm/s)'},**base))
    sampled=df.sample(min(1800,len(df)),random_state=42)
    panel('Relação entre temperatura, vibração e falhas',px.scatter(sampled,x='temperature_c',y='vibration_mm_s',color=sampled.failure.map({0:'Sem falha',1:'Falha'}),opacity=.55,labels={'temperature_c':'Temperatura (°C)','vibration_mm_s':'Vibração (mm/s)','color':'Evento'},**base),390)
elif page=='Análise de falhas':
    headline('Análise de falhas','Eventos registrados, impacto e priorização operacional')
    kpis();st.write('')
    a,b=st.columns(2)
    byeq=df.groupby(['equipment_id','unit'],as_index=False).agg(falhas=('failure','sum'),paradas_h=('downtime_hours','sum')).sort_values('falhas',ascending=False).head(12)
    with a:panel('Equipamentos com mais falhas',px.bar(byeq.sort_values('falhas'),x='falhas',y='equipment_id',orientation='h',labels={'falhas':'Falhas','equipment_id':'Ativo'},**base),390)
    with b:panel('Horas de parada por equipamento',px.bar(byeq.sort_values('paradas_h'),x='paradas_h',y='equipment_id',orientation='h',labels={'paradas_h':'Horas','equipment_id':'Ativo'},**base),390)
    st.markdown('#### Priorização de ativos');st.dataframe(byeq,use_container_width=True,hide_index=True)
elif page=='Manutenção preditiva':
    headline('Manutenção preditiva','Classificação experimental de risco de falha com Random Forest')
    st.info('Modelo demonstrativo treinado com dados sintéticos. Os resultados não representam confiabilidade de equipamentos reais nem substituem avaliação de engenharia.')
    features=['temperature_c','vibration_mm_s','pressure_bar','load_pct','operating_hours','age_years']
    # Train on the full synthetic dataset; split chronologically to reduce future-to-past leakage.
    all_data=raw.sort_values('date').reset_index(drop=True)
    cutoff=int(len(all_data)*.8)
    train,test=all_data.iloc[:cutoff],all_data.iloc[cutoff:]
    @st.cache_resource
    def train_model():
        m=RandomForestClassifier(n_estimators=100,max_depth=9,min_samples_leaf=8,class_weight='balanced',random_state=42,n_jobs=-1)
        m.fit(train[features],train.failure)
        pred=m.predict_proba(test[features])[:,1]
        return m,roc_auc_score(test.failure,pred),precision_score(test.failure,pred>=.5,zero_division=0),recall_score(test.failure,pred>=.5,zero_division=0)
    model,auc,precision,recall=train_model()
    a,b,c=st.columns(3);a.metric('ROC-AUC (teste)',f'{auc:.3f}');b.metric('Precisão (limiar 0,50)',f'{precision:.2%}');c.metric('Recall (limiar 0,50)',f'{recall:.2%}')
    st.markdown('#### Simulador de risco')
    with st.form('prediction'):
        a,b,c=st.columns(3)
        with a: temp=st.number_input('Temperatura (°C)',value=75.0);vib=st.number_input('Vibração (mm/s)',value=3.5)
        with b: pressure=st.number_input('Pressão (bar)',value=6.2);load=st.number_input('Carga (%)',min_value=0.0,max_value=100.0,value=75.0)
        with c: hours=st.number_input('Horas acumuladas',min_value=0.0,value=4000.0);age=st.number_input('Idade do equipamento (anos)',min_value=0,max_value=50,value=7)
        submitted=st.form_submit_button('Estimar risco',type='primary')
    if submitted:
        x=pd.DataFrame([[temp,vib,pressure,load,hours,age]],columns=features)
        score=model.predict_proba(x)[0,1]
        st.metric('Probabilidade estimada de falha no registro',f'{score:.1%}')
        st.caption('Estimativa estatística sobre um evento sintético; não é previsão de falha em horizonte temporal definido.')
    importance=pd.DataFrame({'Variável':features,'Importância':model.feature_importances_}).sort_values('Importância')
    panel('Importância das variáveis no modelo',px.bar(importance,x='Importância',y='Variável',orientation='h',**base))
else:
    headline('Exploração de dados','Consulte os registros sintéticos filtrados e exporte para análises externas')
    st.markdown('#### Amostra dos dados');st.dataframe(df.head(1000),use_container_width=True,hide_index=True)
    st.download_button('Exportar CSV filtrado',df.to_csv(index=False).encode('utf-8-sig'),'industrial_intelligence_filtrado.csv','text/csv',type='primary')
    st.markdown('#### Estrutura do banco de dados');st.code('equipment (equipment_id, unit, type, age_years)\nsensor_readings (reading_id, date, equipment_id, temperature_c, vibration_mm_s, pressure_bar, load_pct, operating_hours, failure, downtime_hours)',language='sql')
st.divider();st.caption('Industrial Intelligence · Python · Pandas · Plotly · Streamlit · SQLite · Scikit-learn | Dados 100% sintéticos')
