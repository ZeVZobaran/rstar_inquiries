# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 12:24:46 2026

@author: José
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima import model
from scipy.optimize import brute
from tabulate import tabulate

# dados
path = path = r'D:\Economia\Agenda Pesquisa\Rstar_Inquiries'
dados_bcb = pd.read_excel(f'{path}/Data/BCB_f.xlsx', index_col=[0])

#%%
# autoarima impondo I=0

def objfunc(order, endog):
    fit = model.ARIMA(endog, None, order).fit()  # none exog
    return fit.bic

def extract_values(best_model):
    const, err = best_model.params[0], best_model.params[-1]*100
    try:
        ar_1 = best_model.params[1]
    except IndexError:
        const = '-'
        ar_1 = '-'
    if len(best_model.params) > 3:
        ar_2 = best_model.params[2]
    else:
        ar_2 = "-"
    return const, ar_1, ar_2, err
   

grid_d0 = (slice(0, 7, 1), #p 0 a 6 de 1 em 1
        slice(0, 1, 1), #d 0
        slice(0, 7, 1)) #q de 0 a 6 de 1 em 1
grid_d2 = (slice(0, 7, 1), #p 0 a 6 de 1 em 1
        slice(0, 3, 1), #d indo até 2
        slice(0, 7, 1)) #q de 0 a 6 de 1 em 1


results = {}
for col in dados_bcb:
    endog = dados_bcb.loc[:, col].dropna().values
    # impondo i=0
    order_messy, bic_i0, gridx, gridf = brute(
        objfunc, grid_d0, args=(endog, ), finish=None, full_output=True
        )
    order_i0 = [int(x) for x in order_messy]
    best_model_i0 = model.ARIMA(endog, None, order_i0).fit()
    const_i0, ar_1_i0, ar_2_i0, err_i0 = extract_values(best_model_i0)
    str_order_i0 = "(" + str(order_i0)[1:-1] + ")"
    # não impondo
    order_messy2, bic_i2, gridx, gridf = brute(
        objfunc, grid_d2, args=(endog, ), finish=None, full_output=True
        )
    order_i2 = [int(x) for x in order_messy2]
    str_order_i2 = "(" + str(order_i2)[1:-1] + ")"
    best_model_i2 = model.ARIMA(endog, None, order_i2).fit()
    const_i2, ar_1_i2, ar_2_i2, err_i2 = extract_values(best_model_i2)

    results[col] = [str_order_i0,
                    const_i0, ar_1_i0, ar_2_i0, err_i0,
                    str_order_i2,
                    const_i2, ar_1_i2, ar_2_i2, err_i2
                    ]

# %%
colunas = ['Ordem', 'C', 'AR(1)', 'AR(2)', '\\sigma^2 \\dot 10^2', 'BIC']

df_results = pd.DataFrame(results)
df_restrito = df_results.iloc[:6, :].T
df_restrito.columns = colunas

df_irr = df_results.iloc[6:, :].T
df_irr.columns = colunas
df_irr = df_irr.reset_index().drop(['index'], axis=1) # indexação pra display

# %%
rest_table_tex = tabulate(
    df_restrito,
    headers=["Especificação", *df_restrito.columns],   # first col is the ratio (index)
    tablefmt="latex_booktabs",
    missingval="--",
    floatfmt=".1f",
    numalign="center",
    stralign='center'
)

irr_table_tex = tabulate(
    df_irr,
    headers=["", *df_irr.columns],   # first col is the ratio (index)
    tablefmt="latex_booktabs",
    missingval="--",
    floatfmt=".1f",
    numalign="center",
    stralign='center'
)


with open(f"{path}/figs/arma_decomp_res.tex", "w", encoding="utf-8") as f:
    f.write(rest_table_tex)

with open(f"{path}/figs/arma_decomp_irr.tex", "w", encoding="utf-8") as f:
    f.write(irr_table_tex)
