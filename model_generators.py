from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import optuna
import pandas as pd
import numpy as np
import prophet
import xgboost
from typing import List

def criar_modelos_fipe(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_train_prophet: pd.DataFrame,
        df_test_prophet: pd.DataFrame,
        exog_train: pd.DataFrame,
        exog_test: pd.DataFrame,
        n_trials: int,
        metrica_erro: str,
        tolerancia_fipe: float,
        target: str
    ) -> pd.DataFrame:

    _, info_fipe_ets, best_value_fipe_ets = generate_ets_model(
        df_train[target],
        df_test[target],
        n_trials,
        metrica_erro,
        tolerancia_fipe
    )

    _, info_fipe_sarimax, best_value_fipe_sarimax = generate_sarimax_model(
        df_train[target],
        df_test[target],
        exog_train,
        exog_test,
        n_trials,
        metrica_erro,
        tolerancia_fipe
    )

    _, info_fipe_prophet, best_value_fipe_prophet = generate_prophet_model(
        df_train_prophet,
        df_test_prophet,
        exog_train.columns.to_list(),
        n_trials,
        metrica_erro,
        tolerancia_fipe
    )

    dict_retorno = {
        'info_fipe_ets': info_fipe_ets,
        'best_value_fipe_ets': best_value_fipe_ets,
        'info_fipe_sarimax': info_fipe_sarimax,
        'best_value_fipe_sarimax': best_value_fipe_sarimax,
        'info_fipe_prophet': info_fipe_prophet,
        'best_value_fipe_prophet': best_value_fipe_prophet
    }

    return dict_retorno

def criar_modelo_final_ets(info: dict, df_train: pd.DataFrame, df_test: pd.DataFrame, horizonte_previsao: int, target: str):
    seasonal_periods = None
    if info['seasonal']:
        seasonal_periods = 12

    damped_trend = info['damped_trend']

    if info['trend'] is None:
        damped_trend = False

    model = ETSModel(
        pd.concat([df_train[target], df_test[target]]),
        error=info['error'],
        trend=info['trend'],
        damped_trend=damped_trend,
        seasonal=info['seasonal'],
        seasonal_periods=seasonal_periods
    )

    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=horizonte_previsao)
    forecast = forecast.rename(target)

    return forecast

def criar_modelo_final_prophet(
        info: dict,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        horizonte_previsao: int,
        target: str,
        df_exog_previsao: pd.DataFrame | None = None,
    ) -> pd.Series:
    model = prophet.Prophet(**info)
    model.fit(pd.concat([df_train, df_test]))  
    future = model.make_future_dataframe(periods=horizonte_previsao, freq='MS', include_history=False)

    if df_exog_previsao is not None:
        future = future.merge(df_exog_previsao, left_on="ds", right_index=True, how="left")

    forecast = model.predict(
        future
    )

    forecast.index = forecast['ds']
    forecast = forecast['yhat']
    forecast = forecast.rename(target)

    return forecast

def criar_modelo_final_sarimax(
        info: dict,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        horizonte_previsao: int,
        target: str,
        df_exog_train: pd.DataFrame | None = None,
        df_exog_test: pd.DataFrame | None = None,
        df_exog_previsao: pd.DataFrame | None = None
    ) -> pd.Series:
    seasonal_order = (0, 0, 0, 0)

    if info['seasonal']:
        seasonal_order = (
            info['P'],
            info['D'],
            info['Q'],
            12
        )

    model = SARIMAX(
        pd.concat([df_train[target], df_test[target]]),
        exog=(
            pd.concat([df_exog_train, df_exog_test])
            if df_exog_train is not None and df_exog_test is not None
            else None
        ),
        order=(
            info['p'],
            info['d'],
            info['q']
        ),
        seasonal_order=seasonal_order,
        trend=info['trend'],
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=horizonte_previsao, exog=df_exog_previsao if df_exog_previsao is not None else None)
    forecast = forecast.rename(target)

    return forecast

def feature_engineering(df: pd.DataFrame, target: str) -> List[str]:
    new_features = []
    n_lags = 3
    
    for i in range(n_lags):
        lag_name = f'lag_{i+1}'
        diff_name = f'diff_{i+1}'
        perc_name = f'perc_{i+1}'

        df[lag_name] = df[target].shift(i+1)
        df[diff_name] = df[target] - df[lag_name]
        df[perc_name] = df[diff_name] / df[target]

        new_features.extend([lag_name, diff_name, perc_name])
    
    mm_name = 'mm_3'
    std_name = 'std_3'
    cv_name = 'cv_3'

    df[mm_name] = df[target].rolling(3).mean()
    df[std_name] = df[target].rolling(3).std()
    df[cv_name] = df[std_name] / df[mm_name]

    new_features.extend([mm_name, std_name, cv_name])

    df.dropna(inplace=True)

    return new_features
    
def generate_prophet_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exog: list[str],
    n_trials: int,
    method: str,
    tol: float = 0
):
    """
    Otimiza os hiperparâmetros de um modelo Prophet utilizando Optuna.

    Para cada trial, um conjunto de hiperparâmetros é amostrado,
    o modelo é treinado com os dados de treino e avaliado sobre o
    conjunto de teste utilizando MAE ou MAPE ponderado.

    Parameters
    ----------
    train : pd.DataFrame
        DataFrame de treinamento contendo as colunas 'ds', 'y' e,
        opcionalmente, as variáveis exógenas.

    test : pd.DataFrame
        DataFrame de teste contendo as colunas 'ds', 'y' e,
        opcionalmente, as variáveis exógenas.

    exog : list[str]
        Lista com os nomes das variáveis exógenas utilizadas pelo modelo.

    n_trials : int
        Número máximo de avaliações realizadas pelo Optuna.

    method : str
        Métrica utilizada na otimização.
        Valores aceitos: 'mae' ou 'mape'.

    tol : float, default=0
        Valor alvo da métrica. Caso seja atingido ou superado,
        o processo de otimização é interrompido antecipadamente.

    Returns
    -------
    tuple[prophet.Prophet, dict, float]
        Tupla contendo:

        - Modelo Prophet configurado com os melhores hiperparâmetros.
        - Dicionário com os melhores hiperparâmetros encontrados.
        - Melhor valor da métrica obtido durante a otimização.
    """
    def objective(trial):

        params = {
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale",
                0.001,
                0.5,
                log=True
            ),

            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale",
                0.01,
                20,
                log=True
            ),

            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode",
                ["additive", "multiplicative"]
            ),

            "changepoint_range": trial.suggest_float(
                "changepoint_range",
                0.6,
                0.95
            ),

            "n_changepoints": trial.suggest_int(
                "n_changepoints",
                5,
                50
            )
        }

        try:
            model = prophet.Prophet(**params)

            model.fit(
                train[['ds', 'y'] + exog]
            )


            forecast = model.predict(
                test[['ds'] + exog]
            )
            
            weights = np.arange(1, len(test)+1)
            if method == 'mape':
                metric = mean_absolute_percentage_error(
                    test['y'],
                    forecast['yhat'],
                    sample_weight=weights    
                )
            else:
                metric = mean_absolute_error(
                    test['y'],
                    forecast['yhat'],
                    sample_weight=weights
                )

            if metric <= tol:
                trial.study.stop()

            return metric
        except Exception as e:
            print(e)
            return np.inf

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False
    )


    best_params = study.best_params
    model_params = {
        k:v
        for k,v in best_params.items()
    }

    model = prophet.Prophet(**model_params)
    return (
        model,
        model_params,
        study.best_value
    )

def generate_ets_model(train: pd.Series, test: pd.Series, n_trials: int, method: str, tol: float = 0):   
    """
    Otimiza os hiperparâmetros de um modelo ETS utilizando Optuna.

    O modelo é avaliado sobre o conjunto de teste utilizando previsões
    para todo o horizonte e métricas ponderadas por posição temporal.

    Parameters
    ----------
    train : pd.Series
        Série temporal utilizada para treinamento.

    test : pd.Series
        Série temporal utilizada para validação.

    n_trials : int
        Número máximo de avaliações realizadas pelo Optuna.

    method : str
        Métrica utilizada na otimização.
        Valores aceitos: 'mae' ou 'mape'.

    tol : float, default=0
        Valor alvo da métrica. Caso seja atingido ou superado,
        o processo de otimização é interrompido antecipadamente.

    Returns
    -------
    tuple[ETSModel, dict, float]
        Tupla contendo:

        - Modelo ETS configurado com os melhores hiperparâmetros.
        - Dicionário com os melhores hiperparâmetros encontrados.
        - Melhor valor da métrica obtido durante a otimização.
    """ 
    def objective(trial):

        error = trial.suggest_categorical("error", ["add", "mul"])
        trend = trial.suggest_categorical("trend", [None, "add", "mul"])
        damped_trend = trial.suggest_categorical("damped_trend", [False, True])

        seasonal = trial.suggest_categorical("seasonal", [None, "add", "mul"])

        seasonal_periods = None
        if seasonal is not None:
            seasonal_periods = 12

        try:
            model = ETSModel(
                train,
                error=error,
                trend=trend,
                damped_trend=damped_trend if trend is not None else False,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )

            fit = model.fit(disp=False)

            forecast = fit.forecast(len(test))

            weights = np.arange(len(test), 0, -1)
            if method == 'mape':
                metric = mean_absolute_percentage_error(
                    test,
                    forecast,
                    sample_weight=weights
                )
            else:
                metric = mean_absolute_error(
                    test,
                    forecast,
                    sample_weight=weights
                )

            if metric <= tol:
                trial.study.stop()

            return metric

        except Exception as e:
            print(e)
            return np.inf

    study = optuna.create_study(direction='minimize')

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False
    )

    best_value = study.best_value
    params = study.best_params

    seasonal_periods = None
    if params['seasonal']:
        seasonal_periods = 12

    model = ETSModel(
        train,
        error=params['error'],
        trend=params['trend'],
        damped_trend=params['damped_trend']
        if params['trend'] is not None
        else False,
        seasonal=params['seasonal'],
        seasonal_periods=seasonal_periods
    )

    return (model, params, best_value)

def generate_sarimax_model(train: pd.Series, test: pd.Series, train_exog: pd.DataFrame | None, test_exog: pd.DataFrame | None, n_trials: int, method: str, tol: float = 0):
    """
    Otimiza os hiperparâmetros de um modelo SARIMAX utilizando Optuna.

    O processo realiza busca sobre os parâmetros não sazonais,
    sazonais e de tendência do modelo, avaliando o desempenho
    das previsões sobre o conjunto de teste.

    Parameters
    ----------
    train : pd.Series
        Série temporal utilizada para treinamento.

    test : pd.Series
        Série temporal utilizada para validação.

    train_exog : pd.DataFrame | None
        Variáveis exógenas utilizadas durante o treinamento.
        Pode ser None caso o modelo não utilize regressoras.

    test_exog : pd.DataFrame | None
        Variáveis exógenas correspondentes ao horizonte de teste.
        Pode ser None caso o modelo não utilize regressoras.

    n_trials : int
        Número máximo de avaliações realizadas pelo Optuna.

    method : str
        Métrica utilizada na otimização.
        Valores aceitos: 'mae' ou 'mape'.

    tol : float, default=0
        Valor alvo da métrica. Caso seja atingido ou superado,
        o processo de otimização é interrompido antecipadamente.

    Returns
    -------
    tuple[SARIMAX, dict, float]
        Tupla contendo:

        - Modelo SARIMAX configurado com os melhores hiperparâmetros.
        - Dicionário com os melhores hiperparâmetros encontrados.
        - Melhor valor da métrica obtido durante a otimização.
    """

    def objective(trial):
        p = trial.suggest_int('p', 0, 5)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 5)

        trend = trial.suggest_categorical(
            'trend',
            ['n', 'c', 't', 'ct']
        )

        seasonal = trial.suggest_categorical(
            'seasonal',
            [False, True]
        )

        seasonal_order = (0, 0, 0, 0)

        if seasonal:
            P = trial.suggest_int('P', 0, 2)
            D = trial.suggest_int('D', 0, 1)
            Q = trial.suggest_int('Q', 0, 2)

            # mensal
            s = 12

            seasonal_order = (P, D, Q, s)

        try:
            model = SARIMAX(
                train,
                exog=train_exog,
                order=(p, d, q),
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            results = model.fit(disp=False)

            forecast = results.forecast(steps=len(test), exog=test_exog)

            weights = np.arange(len(test), 0, -1)
            if method == 'mape':
                metric = mean_absolute_percentage_error(
                    test,
                    forecast,
                    sample_weight=weights
                )
            else:
                metric = mean_absolute_error(
                    test,
                    forecast,
                    sample_weight=weights
                )

            if metric <= tol:
                trial.study.stop()

            return metric

        except Exception as e:
            print(e)
            return float('inf')

    study = optuna.create_study(direction='minimize')

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False
    )

    best_value = study.best_value
    params = study.best_params

    seasonal_order = (0, 0, 0, 0)

    if params['seasonal']:
        seasonal_order = (
            params['P'],
            params['D'],
            params['Q'],
            12
        )

    model = SARIMAX(
        train,
        train_exog,
        order=(
            params['p'],
            params['d'],
            params['q']
        ),
        seasonal_order=seasonal_order,
        trend=params['trend'],
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return (model, params, best_value)