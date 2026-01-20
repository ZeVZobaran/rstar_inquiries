# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 18:21:40 2025

@author: José
"""

import pandas as pd
import numpy as np
from scipy.special import fdtri, fdtr
from pmdarima import arima
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate
import sys
sys.path.append(r'D:\python_utils')
import graphers
path = r'D:\Economia\Agenda Pesquisa\Rstar_Inquiries\Data'

#%% load dados

data = pd.read_excel(f'{path}/BCB_f.xlsx', index_col=[0])
data_clean = data.dropna()

macro_data = pd.read_excel(f'{path}/Data/data_macro.xlsx', index_col=[0],
                            sheet_name='format').dropna()
macro_data_idx = macro_data.loc[data_clean.index]

#%% 1. Testamos para raiz unitária

def test_adf_bunch(df, max_lags=12):
    results = {}
    cte_types = {'constante': 'c', 'constante e tendência': 'ct'}
    for col in df.columns:
        results[col] = {}
        serie = df.loc[:, col]
        for name, reg in cte_types.items():
            ad_fuller_result = adfuller(serie.values, max_lags, reg)
            p_val = ad_fuller_result[1]
            results[col][name] = p_val
    results_df = pd.DataFrame(results)
    return results_df

def test_pp_bunch(df):
    results = {}
    df=data_clean
    for col in df.columns:
        results[col] = {}
        serie = df.loc[:, col]
        pp_result = arima.PPTest().is_stationary(serie)
        p_val = pp_result[0]
        results[col] = p_val
    results_df = pd.Series(results)
    return results_df

# Rejeitamos estacionariedade para valores altos!
adf_results = test_adf_bunch(data_clean)
pp_results = test_pp_bunch(data_clean)
unit_root_df = adf_results.copy()
unit_root_df.loc['Phillips-Perron'] = pp_results 
unit_root_df = unit_root_df.T
unit_root_df.columns= ['ADF/Constante', 'ADF/Constante e Tendência', 'Phillips-Perron']
unit_root_df = unit_root_df.T

unit_root_df_clean = unit_root_df.reset_index()
unit_root_df_clean.columns = ['especificação', 'focus', 'ntnb', 'uip', 'gdp_gap', 'model_consistent']
table_tex = tabulate(unit_root_df_clean,
                     headers=['', *unit_root_df_clean.columns],
                     tablefmt='latex_booktabs',
                     floatfmt=".2f",
                     missingval="--"
                     )
with open(f'{path}/unit_root.tex', "w", encoding="utf-8") as f:
    f.write(table_tex)
    
# E..... rejeitamos estacionariedade em todos os modelos estimados
# Como eu posso "resolver" isso? Se uma série é curta, de forma que eu não consigo
# rejeitar estacionariedade, mas eu sei que ela é estacionária, como posso modelar
# ela para impor estacionariedade e gerar um modelo crível?

#%% 2. Estimamos forma ARMA de cada estimação e comparamos elas
# Comprometido pelo resultado anterior

# vamos testar várias composições ARMA para cada juro e comparar:
def arima_anl(df):
    results = {}
    # I claim: apesar de elas deverem ser estacionarias, não são nessa janela
    # porém, nada _deve_ dar errado tirando diff das séries, e isso permite
    # estimação não-viesada. Para os meus objetivos aqui, sem problemas
    for col in df.columns:
        results[col] = {}
        serie = df.loc[:, col]
        model = arima.auto_arima(serie, max_p=12, max_d=2, max_q=6)
        results[col]['params'] = model.params()
        results[col]['std'] = model.bse()
        results[col]['order'] = model.order
    return results

arima_decomp = arima_anl(data_clean)
# grande bagunça, tudo diferente!
# UIP inclusive é I(2).... CAOS
# De mais importante: temos diferenças tanto na ordem quanto no tamanho 
# estimado de choques

# %% 3,
# Agora vamos testar _impondo_ estacionariedade
    #1. qual é a escolhida por BIC
    #2. qual é o choque gerado em cada uma
# consideramos até 4 lags para a parte AR e a parte MA

def auto_arma(serie, max_p, max_q):
    best_bic = -2000
    for p in range(1, max_p+1):
        for q in range(1, max_q+1):
            model = sm.tsa.arima.ARIMA(
                serie, order=(p, 0, q), enforce_stationarity=False) # não sei como é o backend disso aqui e quero que ele use meus dados RAW
            res = model.fit(cov_type='oim')
            bic = res.bic
            if bic > best_bic:
                best_bic = bic
                best_res = res
                best_order = (p, q)
    return best_res, best_order

def arma_anl(df):
    results = {}
    for col in df.columns:
        results[col] = {}
        serie = df.loc[:, col]
        model, order = auto_arma(serie, 8, 8)
        results[col]['params'] = model.params
        results[col]['std'] = model.bse
        results[col]['order'] = order
    return results

arma_decomp = arma_anl(data_clean)

# AGORA SIM CAOS

#%% 4. Estimamos modelos AR-DL para cada estimação. Pedimos a parte DL = 0

# A ideia aqui, de novo, vai ser trabalhar em diferenças para o juro
# Vamos testar que a estrutura de lags seja igual em diff estipulações
# Se não forem, temos evidência de que diferentes formas de estimação
# respondem de formas diferentes ao cenário macro

#1: usar o melhor modelo como versão restrita
#2: usar muitos lags para a versão longa
#3: teste F de exclusão
def _gen_lags(df, lags):
    for lag in range(lags):
        x = df.copy()
        lagged = x.shift(lag+1)
        for col in lagged:
            x.loc[:, f'{col}_{lag+1}'] = lagged.loc[:, col]
    return x


def ar_dl(serie, exogs, lags=4, order='infer'):
    # internamente queremos um teste F das exógenas em agregado
    # primeiro matamos as obs que serão excluidas pelos lags das exogs
    serie = serie.iloc[lags:]
    # 1: roda e reserva o modelo restrito
    if order=='infer':
        m_restrito = arima.auto_arima(serie, max_p=12, max_d=2, max_q=6)
        order = m_restrito.order
    else:
        m_restrito = arima.ARIMA(order).fit(serie, cov_type='oim')

    # 2: roda sob exógenas
    # consideramos lags de até 1 ano para o período atual por default
    lagged_exogs = _gen_lags(exogs, lags).dropna()

    m_dl = arima.ARIMA(order=order).fit(y=serie, X=lagged_exogs, cov_type='oim')

    # 3: teste F
    r2_restrito = (m_restrito.resid()**2).sum()
    r2_completo = (m_dl.resid()**2).sum()
    p2p1 = len(m_dl.params()) - len(m_restrito.params())
    degrees_freedom = serie.size - len(m_dl.params())
    nk = degrees_freedom / p2p1
    f = nk * (r2_restrito - r2_completo)/r2_completo
    p_val = 1-fdtr(p2p1, degrees_freedom, f)
    return p_val


def ecm_ar_dl(serie, exogs, lags=2):
    # vamos implementar um vetor de correção de erro para tentar lidar com
    # a integração. Usamos a mesma ordem AR e de lags em X por preguiça
    df_use = exogs.copy()
    df_use.loc[:, 'y'] = serie
    for col in df_use:
        df_use.loc[:, f'd_{col}'] = df_use.loc[:, col].diff()
        for lag in range(lags):
            df_use.loc[:, f'd_l{lag+1}_{col}'] = df_use.loc[:, f'd_{col}'].shift(lag+1)
    df_use.loc[:, 'l1_y'] = serie.shift(1)

    cols_drop = [x for x in exogs.columns]
    cols_drop.append('y')
    cols_drop.append('d_y')
    X_ecv_dl = df_use.drop(cols_drop, axis=1)
    X_ecv_dl = sm.add_constant(X_ecv_dl)

    restrito_keep = ['l1_y']
    for lag in range(lags):
        restrito_keep.append(f'd_l{lag+1}_y')
    X_ecv_restrito = df_use.loc[:, restrito_keep]
    X_ecv_restrito = sm.add_constant(X_ecv_restrito)

    Y = df_use.loc[:, 'd_y']
    # Roda o restrito
    m_restrito = sm.OLS(Y, X_ecv_restrito, missing='drop').fit(
        cov_type='HAC',  cov_kwds={'maxlags': 4})

    # 2: roda sob exógenas
    # consideramos lags de até 1 ano para o período atual por default
    m_dl = sm.OLS(Y, X_ecv_dl, missing='drop').fit(
        cov_type='HAC',  cov_kwds={'maxlags': 4})

    # 3: teste F
    r2_restrito = (m_restrito.resid**2).sum()
    r2_completo = (m_dl.resid**2).sum()
    p2p1 = len(m_dl.params) - len(m_restrito.params)
    degrees_freedom = m_dl.nobs - len(m_dl.params)
    nk = degrees_freedom / p2p1
    f = nk * (r2_restrito - r2_completo)/r2_completo
    p_val = 1-fdtr(p2p1, degrees_freedom, f)
    return p_val

f_pvals = {'autoarima': {},
           'ecm': {}
           }

for col in data_clean.columns:
    serie = data_clean.loc[:, col]
    f_pvals['autoarima'][col] = ar_dl(serie, macro_data_idx)
    f_pvals['ecm'][col] = ecm_ar_dl(serie, macro_data_idx)
df_f_pvals = pd.DataFrame(f_pvals)
# Em diferenças todos afirmam que as variáveis extra não importam tanto (pval alto)
    
# Usando o ECM eu cheguei em resultados mais interessantes!
# focus, uip, gdp_gap pvals são altos --> parametros extra não importam
# MAS para model consistent e NTNB pval baixo! Parametros extra importam!!

# %% VAR