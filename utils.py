import datetime
from dateutil import relativedelta
import pandas as pd

def coletar_base_skus(data_ref: datetime.datetime, skus: list[int], df_fipe) -> pd.DataFrame:
    df_previsao_list = []

    data_ref = pd.to_datetime('2026-04-01')

    for sku in skus:
        df = df_fipe.query("sku == @sku").copy()
        df = df.set_index('reference_date')

        data_mais_antiga_sku = df.index.min()

        meses_totais = relativedelta(data_ref, data_mais_antiga_sku).years * 12 + relativedelta(data_ref, data_mais_antiga_sku).months

        if meses_totais < 12:
            print(f"SKU: {sku} não tem dados suficientes ({meses_totais} meses apenas)")
            continue

        meses_esperados = pd.date_range(f'{data_mais_antiga_sku.year}-{data_mais_antiga_sku.month}', f'{data_ref.year}-{data_ref.month}', freq='MS')

        df = df.reindex(meses_esperados)

        df['brand_name'] = df['brand_name'].ffill()
        df['model_name'] = df['model_name'].ffill()
        df['year'] = df['year'].ffill()
        df['fuel_name'] = df['fuel_name'].ffill()

        df['brl_price'] = df['brl_price'].interpolate(method='linear')

        df = df.rename_axis('reference_date').reset_index()

        df.index = df['reference_date']
        df['sku'] = df['sku'].ffill()

        df_previsao_list.append(df)

    df_previsao = pd.concat(df_previsao_list, ignore_index=True)

    return df_previsao

def separar_treino_teste_exogena(df_ipca: pd.DataFrame, df_exchange: pd.DataFrame, dt_inicio_teste: datetime.datetime) -> tuple:
    train_ipca = df_ipca[df_ipca.index < dt_inicio_teste]
    test_ipca = df_ipca[df_ipca.index >= dt_inicio_teste]

    train_exchange = df_exchange[df_exchange.index < dt_inicio_teste]
    test_exchange = df_exchange[df_exchange.index >= dt_inicio_teste]

    exog_train = pd.concat([train_ipca['valor'], train_exchange['exchange_rate']], axis=1)
    exog_test = pd.concat([test_ipca['valor'], test_exchange['exchange_rate']], axis=1)

    exog_train['date'] = pd.to_datetime(exog_train.index)
    exog_test['date'] = pd.to_datetime(exog_test.index)

    return (exog_train, exog_test)

def separar_treino_teste_sku(df_sku: pd.DataFrame, dt_inicio_teste: datetime.datetime) -> tuple:
    train_sku = df_sku[df_sku.index < dt_inicio_teste]
    test_sku = df_sku[df_sku.index >= dt_inicio_teste]

    return (train_sku, test_sku)

def criar_dataset_prophet(df_train: pd.DataFrame, df_test: pd.DataFrame, coluna_data: str, coluna_target: str, colunas_exog: list[str]):
    df_prophet_train = (
        df_train
        .rename(
            columns={
                coluna_data: 'ds',
                coluna_target: 'y'
                }
            )
        .reset_index(drop=True)
        [['ds', 'y'] + colunas_exog]
    )

    df_prophet_test = (
        df_test
        .rename(
            columns={
                coluna_data: 'ds',
                coluna_target: 'y'
                }
            )
        .reset_index(drop=True)
        [['ds', 'y'] + colunas_exog]
    )
    
    return (df_prophet_train, df_prophet_test)