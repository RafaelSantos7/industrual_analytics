from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DB = ROOT / 'industrial.db'

def generate():
    rng = np.random.default_rng(42)
    equipment = pd.DataFrame({
        'equipment_id': [f'EQ-{i:03d}' for i in range(1, 81)],
        'unit': rng.choice(['Unidade Norte','Unidade Sul','Unidade Leste','Unidade Oeste'], 80),
        'type': rng.choice(['Bomba','Compressor','Motor','Turbina','Ventilador'],80),
        'age_years': rng.integers(1,16,80),
    })
    n=24000
    ids=rng.integers(0,80,n)
    dates=pd.Timestamp('2025-01-01')+pd.to_timedelta(rng.integers(0,365,n),unit='D')
    hours=rng.uniform(100,9000,n)
    temperature=rng.normal(68,13,n)+equipment.age_years.to_numpy()[ids]*.45
    vibration=np.clip(rng.normal(2.8,1.1,n)+equipment.age_years.to_numpy()[ids]*.065,.2,None)
    pressure=np.clip(rng.normal(6.3,1.2,n),.3,None)
    load=rng.uniform(30,100,n)
    risk_logit=-6.4+.055*(temperature-65)+.72*(vibration-2.5)+.00018*hours+.022*(load-60)
    probability=1/(1+np.exp(-risk_logit))
    failure=rng.binomial(1,probability)
    downtime=np.where(failure==1,rng.uniform(1.5,22,n),0)
    data=pd.DataFrame({'reading_id':np.arange(1,n+1),'date':dates,'equipment_id':equipment.equipment_id.to_numpy()[ids],
      'temperature_c':np.round(temperature,2),'vibration_mm_s':np.round(vibration,2),
      'pressure_bar':np.round(pressure,2),'load_pct':np.round(load,2),'operating_hours':np.round(hours,1),
      'failure':failure,'downtime_hours':np.round(downtime,2)})
    with sqlite3.connect(DB) as conn:
        equipment.to_sql('equipment',conn,if_exists='replace',index=False)
        data.to_sql('sensor_readings',conn,if_exists='replace',index=False)
    print(f'Banco gerado: {DB} | {n:,} leituras | {failure.sum()} eventos de falha')

if __name__=='__main__': generate()
