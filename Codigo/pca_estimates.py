# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 18:00:32 2025

Main source: https://www.bcb.gov.br/content/ri/relatorioinflacao/202306/ri202306b6p.pdf
Data: BCB SGS e box acima

@author: José
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pmdarima.utils import acf
import statsmodels.api as sm
from scipy.stats import shapiro
import itertools
from tabulate import tabulate
import sys
sys.path.append(r'D:\python_utils')
import graphers
path = r'D:\Economia\Agenda Pesquisa\Rstar_Inquiries'

#%% load dados

data = pd.read_excel(f'{path}/Data/BCB_f.xlsx', index_col=[0])
data_clean = data.dropna()
r_bcb = graphers.gen_graf(data_clean, titulo='Métricas de R* do BCB', subtitulo='retirados do RPM 06/24')
r_bcb.show(renderer='png')
correls = data.corr()
correls_texto = [np.round(x, 2) for x in correls.values]
correls_hmp = graphers.gen_heatmap(correls*-1, 'Mapa de Correlações de Juros Neutros do BCB',
                                   midpoint=0, texto=correls_texto, eixo_no_topo=True)

correls_hmp.show(renderer='png')

pca = PCA(n_components=len(data.columns))
data_pca = pca.fit_transform(data_clean)

df_pca = pd.DataFrame(data_pca,
                      columns=[f'pca_{i}' for i in range(len(data.columns))],
                      index=data_clean.index)

# Plot PCA results

nomes = [
    f'fator {i+1} ({round(100*pca.explained_variance_ratio_[i], 1)}%)' 
    for i in range(len(pca.explained_variance_ratio_))
    ]
pca_full = graphers.gen_graf(
    df_pca, titulo='Decomposição PCA dos modelos de R* do BCB',
    subtitulo='% da variância explicada entre parênteses', nomes=nomes)
pca_full.show(renderer='png')
# em 5 especificações identificamos 3 fontes de variância distintas
# idealmente identificaríamos apenas uma!
#%% Erros

diffs = data_clean.values[:, :, None] - data_clean.values[:, None, :]
cols = data_clean.columns

i, j = np.triu_indices(len(cols ), k=1)
diff_df = pd.DataFrame(
    diffs[:, i, j],
    columns=[f'{cols[i[k]]}-{cols[j[k]]}' for k in range(len(i))],
    )

# Quero checar se / quais desses aqui tem cara de resíduo
# média zero, ortogonalidade, normalidade, autocorr comportada
diff_df.mean().abs() < diff_df.std()
diff_df_ac = diff_df.apply(acf)

diff_df_ac.plot()
ac_fig = graphers.gen_graf(
    diff_df_ac, titulo='ACFs das Diferenças')
ac_fig.show(renderer='png')
# Apesar da maior parte das médias serem zero, elas todas tem AC serial altíssima!
# evidência no sentido do que achamos
shapiro_results = diff_df.apply(shapiro).T

# %% Erros em diferenças
d_diffs = data_clean.diff().values[:, :, None] - data_clean.diff().values[:, None, :]
cols = data_clean.columns

i, j = np.triu_indices(len(cols ), k=1)
d_diff_df = pd.DataFrame(
    d_diffs[:, i, j],
    columns=[f'{cols[i[k]]}-{cols[j[k]]}' for k in range(len(i))],
    ).dropna()

d_diff_df.mean().abs() < d_diffs.std()
d_diff_ac = d_diff_df.apply(acf)
dac_fig = graphers.gen_graf(
    d_diff_ac, titulo='ACFs em Diferenças')
dac_fig.show(renderer='png')
# melhor comportadas...
# Testando normalidade
# p_val baixo --> não normal
shapiro_results = d_diff_df.apply(shapiro).T
# Em geral não normalidade tá valendo aqui

# %% Estudando covariância

cov_rstar = data.cov()
# tese aqui: as razões entre as covariâncias não podem ser muito altas
# na medida em que elas forem, isso nos indica que a mensuração de algumas séries
# é bem esquisita
# Como eu tenho sobre-identificação, da pra comparar na pele na real!    
via_focus = cov_rstar.loc['focus', 'ntnb']/cov_rstar.loc['focus', 'uip']
via_mc = cov_rstar.loc['model_consistent', 'ntnb']/cov_rstar.loc['model_consistent', 'uip']
via_gdp_gap = cov_rstar.loc['gdp_gap', 'ntnb']/cov_rstar.loc['gdp_gap', 'uip']

eps = 1e-2
clean_cov = cov_rstar.apply(lambda x: x.apply(lambda y: None if abs(y) < eps else y))

ratios = {}
for num, den in itertools.permutations(clean_cov.columns, 2):
    ratios[f"{num}/{den}"] = clean_cov[num] / clean_cov[den]

ratio_df_full = pd.DataFrame(ratios)
f_vals = []
for row in ratio_df_full.T:
    name = ratio_df_full.loc[row, :].name
    vals = []
    for col in ratio_df_full.loc[[row], :]:
        if name in col:
            vals.append(np.NaN)
        else:
            vals.append(ratio_df_full.loc[row, col])
    f_vals.append(vals)

ratio_df_full = pd.DataFrame(f_vals, columns=ratio_df_full.columns, index=ratio_df_full.index)


ratio_df_clean = ratio_df_full.drop([
    'ntnb/focus', 'uip/focus', 'uip/ntnb', 'gdp_gap/focus', 'gdp_gap/ntnb',
    'gdp_gap/uip', 'model_consistent/focus', 'model_consistent/ntnb',
    'model_consistent/uip', 'model_consistent/gdp_gap'
    ], axis=1).T


def pretty(num):
    if not np.isnan(num):
        result = np.round(num, 1)
    else:
        result = '-'
    return result

ratio_df_clean = ratio_df_clean.apply(lambda x: x.apply(lambda y:
                                                        pretty(y)))


ratio_df_clean = ratio_df_clean.reset_index()
ratio_df_clean.columns = ['razão', 'focus', 'ntnb', 'uip', 'gdp_gap', 'model_consistent']
table_tex = tabulate(ratio_df_clean,
                     headers=['', *ratio_df_clean.columns],
                     tablefmt='latex_booktabs',
                     floatfmt=".1f",
                     missingval="--"
                     )
with open(f'{path}/figs/cov_ratio_table.tex', "w", encoding="utf-8") as f:
    f.write(table_tex)

# %% pegando e formatando dados macro
# Dados formatados
r_mean = data_clean.mean(axis=1)

macro_data = pd.read_excel(f'{path}/Data/data_macro.xlsx', index_col=[0],
                            sheet_name='format').dropna()
# ajustando indíce e impondo lag de um quarter
# melhor para garantir que não estamos usando informação que não estaria
# disponível
macro_data_idx = macro_data.shift().loc[r_mean.index]
lag_macro_data_cte = sm.add_constant(macro_data_idx)

lag_macro_data_cte.plot()

# 1o exercício:
# 1st step: reg R_i em PCA_1. Aff: o resíduo disso é Fi*M(t) + u(t)
# 2nd step: reg res em Mt, identifica Fi
hac_lags = int(round(4*(r_mean.size/100)**(2/9), 0))
pca_1_cte = sm.add_constant(df_pca.pca_0)
models = {
    column:
    sm.OLS(data_clean.loc[:, column], pca_1_cte).fit(cov_type='HAC',
                                                     cov_kwds={'maxlags': hac_lags})
    for column in data_clean
    }

resids = pd.DataFrame(
    {key: value.resid for key, value in models.items()},
    )

# rodamos o modelo sem cte, visto que a média de resids já é zero
l_macro_sem_cte = lag_macro_data_cte.iloc[:, 1:]
resid_models = {
    column:
    sm.OLS(resids.loc[:, column], l_macro_sem_cte).fit(cov_type='HAC',
                                              cov_kwds={'maxlags': hac_lags})
    for column in resids
    }

#%% Análise de objetos 1: betas Ri ~ PCA
betas = pd.Series(
    {key: value.params[1] for key, value in models.items()}
    )
df_coefs = graphers.regtable(models, to_latex=False)

# Chocantemente, os sinais variam. Interpretação:
    #1: PCA1 é algo que é incorporado com sinal negativo em NTNB 
    # e negativo no resto todo
    #2: H3 na veia

model_pca_ex_1 = PCA(n_components=len(resids.columns))
pca_ex_1 = model_pca_ex_1.fit_transform(resids)
nomes = [
    f'fator {i+1} ({round(100*model_pca_ex_1.explained_variance_ratio_[i], 1)}%)' 
    for i in range(len(model_pca_ex_1.explained_variance_ratio_))
    ]
df_pca_ex_1 = pd.DataFrame(pca_ex_1,
                      columns=[f'pca_ex_1{i}' for i in range(len(resids.columns))],
                      index=data_clean.index)

pca_resids = graphers.gen_graf(
    df_pca_ex_1, titulo='Resids PCA',
    subtitulo='% da variância explicada entre parênteses', nomes=nomes)
pca_resids.show(renderer='png')

# Sobram duas fontes de variância bem fortes!
# %% Análise 2: betas do primeiro passo batem com os de covariâncias
ratios_pca = {
    "focus/ntnb": betas[0]/betas[1],
    "focus/uip": betas[0]/betas[2],
    "focus/gdp_gap": betas[0]/betas[3],
    "focus/model_consistent": betas[0]/betas[4],
    "ntnb/uip": betas[1]/betas[2],
    "ntnb/gdp_gap": betas[1]/betas[3],
    "ntnb/model_consistent": betas[1]/betas[4],
    "uip/gdp_gap": betas[2]/betas[3],
    "uip/model_consistent": betas[2]/betas[4],
    "gdp_gap/model_consistent": betas[3]/betas[4],
    }
ratio_df_clean.loc[:, 'PCA'] = pd.Series(ratios_pca).round(1).values
# Ambíguo de novo: talvez algumas tenham loads e outras não?

#%% Teste F de resids_i ~ macro
f_tests = {
    key: {
        'F stat': round(value.fvalue, 2),
        'P value': round(value.f_pvalue, 2),
        "DFs": (int(value.df_model), int(value.df_resid))
        } for key, value in resid_models.items()
    }
f_table = pd.DataFrame(f_tests).T
# and we fail to reject! Bloco macro deve ter poder explicativo, mesmo filtrando
# o R*!

#%%  4: queremos estudar se Fi faz sentido com a forma de estimação usada

df_coefs_resids = graphers.regtable(
    resid_models, to_latex=False
    )
# Focus não dá pra interpretar
# IPCA_YA_GAP pega em todos; embi em todos menos UIP; US rate pega no UIP Lol
# nenhum pega hiato!































