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
from pmdarima.arima import auto_arima
from pmdarima.utils import acf
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.optimize import least_squares
from scipy.stats import shapiro
import sys
sys.path.append(r'D:\python_utils')
import graphers
path = r'D:\Economia\Mestrado EPGE\Rstar Inquiries'

#%% load dados

data = pd.read_excel(f'{path}/BCB_f.xlsx', index_col=[0])
data_clean = data.dropna()
r_bcb = graphers.gen_graf(data_clean, titulo='Métricas de R* do BCB', subtitulo='retirados do RPM 06/24')
r_bcb.show(renderer='png')
correls = data.corr()

pca = PCA(n_components=len(data.columns))
data_pca = pca.fit_transform(data_clean)

pca.explained_variance_ratio_
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
    diff_df_ac, titulo='ACFs em Nível')
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

#%%
# R(t, i) = R*(t) + e(t, i)
# e(t, i) = L(i)*F(t) + u(t)

# vejo Y(t, i), não vejo mais nada nessa joça
# sum(R)/I = R*(t) + u(t) + SUM(L(i))*F(t)
# Adoraria informação sobre SUM(L(i))

# Exatminando o objeto sum(R)/I:

r_mean = data_clean.mean(axis=1)

arima_model = auto_arima(
        r_mean, start_p=0, start_q=0, max_p=6, max_q=6, m=12,
        seasonal=True, start_d=0, max_d=1, D=0, trace=False,
        error_action='ignore', suppress_warnings=True, stepwise=True
        )
arima_model.plot_diagnostics()
arima_model.summary()

# idenfitiquei que d(u(t)) ~ N(0, 0.005)
# e que  d(R*(t)) + SUM(L(i))*d(F(t)) é um AR(1)
# d(F(t)) é um objeto sobre o qual temos informação
# SUM(L(i)) é a soma dos vetores de peso de cada variável em cada erro
# é uma espécie de vetor de betas então
# reescrevendo:
# d(sum(R)/I) = d(R*(t)) + SUM(L(i))*d(F(t)) + d(u(t)) 
# Estimação não paramétrica + OLS pode funcionar? 
# Permitimos que d(R*(t)) seja em qualquer estrutura e impomos a relação linear
# para o loading
# Arrumar um note pra falar com o Raul?
# Não vou identificar a matriz de loading completa, só a soma de todos os modelos
# usados
# MAS devo conseguir estimar o R* subjacente a todas e o tamanho do erro que
# cada variável induz. Como cada modelo "foca" em uma coisa diferente, deve
# ser possível falar qualitativamente sobre isso!!

# Problema nessa forma: R* e F são super correlacionados...
# Argumento a favor dessa forma de estimação: 
# Se todas as variáveis macro ficaram iguais, a estimativa
# não deve acrescentar erro derivado de variáveis macro. Qualquer mudança observada
# deve ser real! Trabalhar em logs me ajuda nesse caso
# especialmente por conta das considerações de gaps 
# um argumento parecido deve valer para nível e gaps, acho?

# %% pegando e formatando dados macro
# Dados formatados
macro_data = pd.read_excel(f'{path}/data_macro.xlsx', index_col=[0],
                            sheet_name='format').dropna()
macro_data_idx = macro_data.loc[r_mean.index]

# 1o exercício:
# 1st step: reg R_mean em PCA_1. Aff: o resíduo disso é F*M(t) + u(t)
# 2nd step: reg res em Mt, identifica ^F
# 3rd step: R_mean - ^F*M = R*
pca_1_cte = sm.add_constant(df_pca.pca_1)
model_res = sm.OLS(r_mean, pca_1_cte)
results_res = model_res.fit(cov_type='HC1')
resid_rmean = results_res.resid

model_resid_noise = sm.OLS(resid_rmean, macro_data_idx )
results_resid_noise = model_resid_noise.fit(cov_type='HC1')
macro_noise = results_resid_noise.fittedvalues
hidden_rstar = r_mean - macro_noise

# Plotting comparisons
df_comparison = pd.DataFrame(r_mean, columns=['mean'])
df_comparison.loc[:, 'hidden_pca'] = hidden_rstar

graf_rstar_pca = graphers.gen_graf(
    df_comparison.loc[:, ['mean', 'noise_pca', 'hidden_pca']], titulo='Juro Neutro Brasileiro Verdadeiro',
    subtitulo='por decomposição two-step PCA OLS',
    nomes=['estimativa do BC', 'viés de modelagem agregado',
           'estimativa limpa de ruído']
    )
graf_rstar_pca.show(renderer='png')

# %%
# Segundo exercício:
# Approach misto OLS e Local Constant NP
# R_mean = B*R_star + F*M + e
# Tratamos B*R_star como m(t), função local de tempo
# 1: reg r_mean ~ t e M ~ t via NW LL. Adquire os erros v, w
# 2: reg v ~ w via OLS. O parâmetro é F
# 3: Reg R_mean - F*M ~ t. O resultado é R*
time_const = np.ones(r_mean.shape[0]).cumsum()

def silverman_bw(x):
    x = np.asanyarray(x).ravel()
    n = x.shape[0]
    std = np.std(x, ddof=1)
    trim = np.subtract(*np.percentile(x, [75, 25]))
    a = min(std, trim/1.34)
    bw = 1.06 * a * n ** (-1/5)
    return bw

def nw_run(col):
    bw=silverman_bw(col)*10
    time_const = np.ones(col.shape[0]).cumsum()
    model = KernelReg(
        endog=np.asanyarray(col),
        exog=time_const,
        var_type='c',
        reg_type='ll',
        bw=[bw]
    )
    return model.fit(time_const)[0]

nw_v = nw_run(r_mean)
v = r_mean - nw_v

macro_filter = macro_data_idx.apply(nw_run)
w = macro_data_idx - macro_filter

ols_step = sm.OLS(v, w).fit(cov_type='HC1')
F_three_step = ols_step.params
denoised = r_mean - macro_data_idx@F_three_step
r_star_three_step = nw_run(denoised)

df_comparison.loc[:, 'hidden_LL'] = r_star_three_step
df_comparison.plot()

# Parece só o resultado de um filtro passa-médias....

# %%
# Terceiro exercício
# A partir do erro identificado no AR de d(R_mean)
# calculei uma log-verossimilhança para identificar os parâmetros
# como eu tenho a variância, acaba que posso usar a CPO nela pra identificar
# um terceiro parâmetro


def log_likelihood_system(guess, r_mean, macro, phi, sigma2, lam):

    T = macro.shape[0]-1
    r_star_diff = pd.Series(
        guess[:T],
        index=macro.index[1:]
        )
    B = guess[T]
    F = guess[T+1:]

    rm_d = r_mean.diff().dropna()
    m_d = macro.diff().dropna()
    m_d_s = m_d.shift().dropna()
    m_d = m_d.loc[m_d_s.index]
    
    r_d = r_star_diff
    r_d_s = r_d.shift().dropna()
    r_d = r_d.loc[r_d_s.index]

    xt = rm_d-phi*(B*r_d_s + (m_d_s @ F))
    xt = xt.dropna()

    # erros
    e_1 = np.array([xt @ (phi*r_d_s)])
    e_2 = np.array(xt @ (phi*m_d_s))
    e_3 = np.array([(xt @ xt)/xt.shape[0] - sigma2])

    # imposição de r* smooth:
    r_dd = r_d.diff()[1:]
    pen = np.array(np.sqrt(lam)*r_dd)
    
    return np.concatenate([e_1, e_2, e_3, pen])

phi, sigma2 = arima_model.params()
lam=1e-2

B_0 = [1]
F_0 = np.ones(macro_data.shape[1])
r_diff_0 = r_mean.diff().dropna()
guess_0 = np.concatenate([r_diff_0, B_0, F_0])
macro = macro_data_idx

res = least_squares(
    log_likelihood_system, guess_0,
    args=(r_mean, macro_data_idx, phi, sigma2, lam),
    method='trf',
    jac='2-point',
    max_nfev=1000, verbose=2,
    bounds=(-1.5, 1.5)
    )

guess_final = res.x
r_diff_final = pd.Series(
    guess_final[:69], r_mean.index[1:]
    )
r_diff_final[1:].plot()
B_final = guess_final[-7]
F_final = guess_final[-6:]

r_est = r_diff_final.cumsum() + r_mean.iloc[0]
df_comparison.loc[:, 'log_l'] = r_est
df_comparison.plot()
# Não adorei não

#%% 4 Equações simultâneas ao invés de log likelihood

def calc_eq_one(r_mean, r_star, macro, u, b):
    # recebe o r_star calculado acima
    # e estima F, e por OLS
    endog = (r_mean.diff().shift(-1) - b*r_star.diff() - u.shift(-1))/b
    endog = endog.dropna()
    X = macro.diff().loc[endog.index]
    model = sm.OLS(endog, X)
    result = model.fit(cov_type='HC1')
    # extraímos o novo F e novo erro
    F = result.params
    d_e = result.resid
    return F, d_e, result

def calc_eq_two(r_mean, F, macro, e):
    r_star = r_mean - macro@F - e
    return r_star
    # geramos um novo r_star


# começamos com um chute para F e para e
# chute: F = 0.1k, e=0n
runs = 0
b = arima_model.params()[0]
u = arima_model.resid()
r_star = pd.Series(
    np.zeros(r_mean.shape[0]),
    index=r_mean.index)
r_star_list=[]
while True:
    F, d_e, model = calc_eq_one(r_mean, r_star, macro_data_idx, u, b)
    d_e.loc[r_mean.index[0]] = 0
    d_e.loc[r_mean.index[-1]] = 0
    d_e = d_e.sort_index()
    e = d_e.cumsum()

    r_star_new = calc_eq_two(r_mean, F, macro_data_idx, e)

    diff = sum((r_star - r_star_new)**2)
    print(diff)

    r_star_list.append(r_star)
    r_star = r_star_new

    if abs(diff) < 0.001:
        break
    elif runs > 100:
        print('too many runs')
        break
    runs += 1

model.summary()
r_star_conv = pd.DataFrame(r_star_list).T
r_star_conv.iloc[:, -5:].plot()

df_comparison.loc[:, 'eqs_simult'] = r_star_new
df_comparison.plot()
# TALVEZ TENHA DADO CERTO?????

