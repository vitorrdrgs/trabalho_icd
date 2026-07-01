import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import model_generators as mg
import numpy as np

def coletar_base_skus(data_ref: datetime.datetime, skus: list[int], df_fipe) -> pd.DataFrame:
    df_previsao_list = []

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
    df_previsao.index = df_previsao['reference_date']

    return df_previsao

def separar_treino_teste_exogena(df_ipca: pd.DataFrame, df_exchange: pd.DataFrame, dt_inicio_teste: datetime.datetime, data_limite_inferior: datetime.datetime) -> tuple:
    train_ipca = df_ipca[(df_ipca.index < dt_inicio_teste) & (df_ipca.index >= data_limite_inferior)]
    test_ipca = df_ipca[df_ipca.index >= dt_inicio_teste]

    train_exchange = df_exchange[(df_exchange.index < dt_inicio_teste) & (df_exchange.index >= data_limite_inferior)]
    test_exchange = df_exchange[df_exchange.index >= dt_inicio_teste]

    exog_train = pd.concat([train_ipca['valor'], train_exchange['exchange_rate']], axis=1)
    exog_test = pd.concat([test_ipca['valor'], test_exchange['exchange_rate']], axis=1)

    exog_train['date'] = pd.to_datetime(exog_train.index)
    exog_test['date'] = pd.to_datetime(exog_test.index)

    return (exog_train, exog_test)

def separar_treino_teste_sku(df_sku: pd.DataFrame, data_ref: datetime.datetime, dt_inicio_teste: datetime.datetime) -> tuple:
    df_train_sku = df_sku[df_sku.index < dt_inicio_teste]
    df_test_sku = df_sku[df_sku.index >= dt_inicio_teste]

    df_train_sku = df_train_sku[df_train_sku['reference_date'] >= f'{data_ref.year - 5}-01-01']

    return (df_train_sku, df_test_sku)

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

def selecionar_modelo_exog(horizonte_previsao: int, **kwargs) -> pd.DataFrame:
    best_value_ipca_prophet = kwargs['best_value_ipca_prophet']
    best_value_ipca_sarimax = kwargs['best_value_ipca_sarimax']
    best_value_exchange_prophet = kwargs['best_value_exchange_prophet']
    best_value_exchange_sarimax = kwargs['best_value_exchange_sarimax']

    info_ipca_prophet = kwargs['info_ipca_prophet']
    info_ipca_sarimax = kwargs['info_ipca_sarimax']
    info_exchange_prophet = kwargs['info_exchange_prophet']
    info_exchange_sarimax = kwargs['info_exchange_sarimax']

    df_prophet_train_ipca = kwargs['df_prophet_train_ipca']
    df_prophet_test_ipca = kwargs['df_prophet_test_ipca']
    df_prophet_train_exchange = kwargs['df_prophet_train_exchange']
    df_prophet_test_exchange = kwargs['df_prophet_test_exchange']

    df_exog_train = kwargs['df_exog_train']
    df_exog_test = kwargs['df_exog_test']

    metricas_ipca = np.array([best_value_ipca_prophet, best_value_ipca_sarimax])
    metricas_exchange = np.array([best_value_exchange_prophet, best_value_exchange_sarimax])

    idx_ipca = np.argmin(metricas_ipca)
    idx_exchange = np.argmin(metricas_exchange)

    forecast_ipca = pd.Series()
    forecast_exchange = pd.Series()
    modelo_escolhido_ipca = ''
    modelo_escolhido_exchange = ''

    if idx_ipca == 0:
        modelo_escolhido_ipca = 'PROPHET'
        forecast_ipca = mg.criar_modelo_final_prophet(
            info_ipca_prophet,
            df_prophet_train_ipca,
            df_prophet_test_ipca,
            horizonte_previsao,
            'valor'
        )
    elif idx_ipca == 1:
        modelo_escolhido_ipca = 'SARIMAX'
        forecast_ipca = mg.criar_modelo_final_sarimax(
            info_ipca_sarimax,
            df_exog_train,
            df_exog_test,
            horizonte_previsao,
            'valor'
        )

    if idx_exchange == 0:
        modelo_escolhido_exchange = 'PROPHET'
        forecast_exchange = mg.criar_modelo_final_prophet(
            info_exchange_prophet,
            df_prophet_train_exchange,
            df_prophet_test_exchange,
            horizonte_previsao,
            'exchange_rate'
        )
    elif idx_exchange == 1:
        modelo_escolhido_exchange = 'SARIMAX'
        forecast_exchange = mg.criar_modelo_final_sarimax(
            info_exchange_sarimax,
            df_exog_train,
            df_exog_test,
            horizonte_previsao,
            'exchange_rate'
        )

    forecast_exchange = forecast_exchange.rename("exchange_rate")
    forecast_ipca = forecast_ipca.rename("valor")

    exog_previsao = pd.concat([forecast_ipca, forecast_exchange], axis=1)

    return exog_previsao, modelo_escolhido_ipca, modelo_escolhido_exchange

def selecionar_modelo_fipe(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_train_prophet: pd.DataFrame,
        df_test_prophet: pd.DataFrame,
        df_exog_train: pd.DataFrame,
        df_exog_test: pd.DataFrame,
        df_exog_previsao: pd.DataFrame,
        horizonte_previsao: int,
        target: str,
        **kwargs
    ) -> tuple:
    best_value_fipe_ets = kwargs['best_value_fipe_ets']
    best_value_fipe_sarimax = kwargs['best_value_fipe_sarimax']
    best_value_fipe_prophet = kwargs['best_value_fipe_prophet']
    best_value_fipe_xgboost = kwargs['best_value_fipe_xgboost']

    info_fipe_ets = kwargs['info_fipe_ets']
    info_fipe_sarimax = kwargs['info_fipe_sarimax']
    info_fipe_prophet = kwargs['info_fipe_prophet']
    
    metricas = np.array([best_value_fipe_ets, best_value_fipe_sarimax, best_value_fipe_prophet, best_value_fipe_xgboost])

    idx_metricas = np.argmin(metricas)

    forecast = pd.Series()
    modelo_escolhido = ''

    if idx_metricas == 0:
        modelo_escolhido = 'ETS'
        forecast = mg.criar_modelo_final_ets(
            info_fipe_ets,
            df_train,
            df_test,
            horizonte_previsao,
            target
        )
    elif idx_metricas == 1:
        modelo_escolhido = 'SARIMAX'
        forecast = mg.criar_modelo_final_sarimax(
            info_fipe_sarimax,
            df_train,
            df_test,
            horizonte_previsao,
            target,
            df_exog_train,
            df_exog_test,
            df_exog_previsao
        )
    elif idx_metricas == 2:
        modelo_escolhido = 'PROPHET'
        forecast = mg.criar_modelo_final_prophet(
            info_fipe_prophet,
            df_train_prophet,
            df_test_prophet,
            horizonte_previsao,
            target,
            df_exog_previsao
        )
    elif idx_metricas == 3:
        modelo_escolhido = 'XGBOOST'
        modelo_fipe_xgboost = kwargs['modelo_fipe_xgboost']
        df_train_xgboost = df_train.join(df_exog_train, how='inner')
        df_test_xgboost = df_test.join(df_exog_test, how='inner')

        forecast = mg.prever_futuro_recursivo(
            modelo_fipe_xgboost,
            pd.concat([df_train_xgboost, df_test_xgboost]),
            df_exog_previsao,
            horizonte_previsao,
            target,
            df_exog_train.columns.tolist()
        )

    forecast = forecast.rename(target)

    return forecast, modelo_escolhido
