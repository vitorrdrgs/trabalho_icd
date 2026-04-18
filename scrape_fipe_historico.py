"""
O script:
1) busca as tabelas de referência (meses disponíveis);
2) para cada referência, coleta marcas;
3) para cada marca, coleta modelos;
4) para cada modelo, coleta anos/modelos;
5) para cada ano/modelo, consulta o valor FIPE completo;
6) salva incrementalmente em CSV a cada marca rodada;

Exemplo limitado a algumas marcas:
    python scrape_fipe_historico.py --start 2019-01 --end 2026-04 --vehicle carros --brands "Toyota,Ford,Fiat,GM - Chevrolet,VW - VolksWagen,Citroën,Honda,Hyundai,Nissan,Peugeot,Renault"

Saída:
    data/raw/fipe_historico_carros.csv

Dependências:
    pip install requests pandas tenacity
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests import Session
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

BASE_URL = "https://veiculos.fipe.org.br/api/veiculos"
ENDPOINTS = {
    "referencias": f"{BASE_URL}/ConsultarTabelaDeReferencia",
    "marcas": f"{BASE_URL}/ConsultarMarcas",
    "modelos": f"{BASE_URL}/ConsultarModelos",
    "ano_modelo": f"{BASE_URL}/ConsultarAnoModelo",
    "valor": f"{BASE_URL}/ConsultarValorComTodosParametros",
}

MONTHS_PT = {
    "janeiro": 1,
    "fevereiro": 2,
    "março": 3,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

VEHICLE_TYPES = {
    "carros": 1,
    "motos": 2,
    "caminhoes": 3,
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/json; charset=UTF-8",
    "Origin": "https://veiculos.fipe.org.br",
    "Referer": "https://veiculos.fipe.org.br/",
    "X-Requested-With": "XMLHttpRequest",
}


class FipeHTTPError(Exception):
    pass


class FipeParseError(Exception):
    pass


@dataclass
class ReferenceMonth:
    codigo_tabela_referencia: int
    mes_raw: str
    ano: int
    mes_num: int
    ym: str  # YYYY-MM

    @property
    def date_str(self) -> str:
        return f"{self.ano:04d}-{self.mes_num:02d}-01"


@dataclass
class FipeRow:
    reference_code: int
    reference_month: str
    reference_date: str
    vehicle_type: int
    brand_code: str
    brand_name: str
    model_code: str
    model_name: str
    year_model_code: str
    ano: int
    fuel_code: int
    fuel_name: str
    codigo_fipe: str
    autenticacao: str
    preco_raw: str
    preco_brl: float
    mes_referencia_api: str
    tipo_veiculo: int
    sigla_combustivel: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def parse_brl(value: str) -> float:
    # Ex.: "R$ 53.296,00"
    if value is None:
        return float("nan")
    s = str(value)
    s = s.replace("R$", "").replace(".", "").replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_reference_month(raw: str, code: Any) -> ReferenceMonth:
    """
    Espera algo como:
      "abril de 2024"
      "julho/2018 "
    """
    s = normalize_text(raw).lower()

    # caso 1: "abril de 2024"
    m = re.match(r"([a-zçãõéíáú]+)\s+de\s+(\d{4})", s)
    if m:
        month_name, year = m.groups()
        month_num = MONTHS_PT.get(month_name)
        if not month_num:
            raise FipeParseError(f"Mês inválido: {raw}")
        return ReferenceMonth(
            codigo_tabela_referencia=int(code),
            mes_raw=raw,
            ano=int(year),
            mes_num=month_num,
            ym=f"{int(year):04d}-{month_num:02d}",
        )

    # caso 2: "julho/2018"
    m = re.match(r"([a-zçãõéíáú]+)/(\d{4})", s)
    if m:
        month_name, year = m.groups()
        month_num = MONTHS_PT.get(month_name)
        if not month_num:
            raise FipeParseError(f"Mês inválido: {raw}")
        return ReferenceMonth(
            codigo_tabela_referencia=int(code),
            mes_raw=raw,
            ano=int(year),
            mes_num=month_num,
            ym=f"{int(year):04d}-{month_num:02d}",
        )

    raise FipeParseError(f"Formato de mês de referência não reconhecido: {raw}")


def parse_year_model_code(code: str) -> Tuple[int, int]:
    """
    Ex.: "2014-1" -> (2014, 1)
    onde 1 costuma indicar combustível/tipo específico.
    """
    s = str(code).strip()
    m = re.match(r"(\d{4})-(\d+)", s)
    if not m:
        raise FipeParseError(f"year_model_code inválido: {code}")
    ano, fuel_code = m.groups()
    return int(ano), int(fuel_code)


def append_rows_csv(path: str, rows: List[FipeRow]) -> None:
    if not rows:
        return
    df = pd.DataFrame([asdict(r) for r in rows])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8-sig")


def create_session(timeout: int = 30) -> Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    session.timeout = timeout  # atributo auxiliar
    return session


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException, FipeHTTPError)),
    reraise=True,
)
def post_json(session: Session, url: str, payload: Optional[Dict[str, Any]] = None) -> Any:
    resp = session.post(url, json=payload or {}, timeout=getattr(session, "timeout", 30))
    if resp.status_code >= 400:
        raise FipeHTTPError(f"HTTP {resp.status_code} em {url}: {resp.text[:300]}")
    try:
        return resp.json()
    except Exception as e:
        raise FipeHTTPError(f"Resposta não-JSON em {url}: {str(e)} | body={resp.text[:300]}")


def get_references(session: Session) -> List[ReferenceMonth]:
    data = post_json(session, ENDPOINTS["referencias"], {})
    refs: List[ReferenceMonth] = []
    for item in data:
        code = item.get("Codigo") or item.get("codigo")
        mes = item.get("Mes") or item.get("mes") or item.get("Month")
        if code is None or mes is None:
            continue
        try:
            refs.append(parse_reference_month(mes, code))
        except FipeParseError:
            continue
    refs.sort(key=lambda x: (x.ano, x.mes_num))
    return refs


def filter_references(refs: List[ReferenceMonth], start: str, end: str) -> List[ReferenceMonth]:
    return [r for r in refs if start <= r.ym <= end]


def get_marcas(session: Session, codigo_tabela_referencia: int, codigo_tipo_veiculo: int) -> List[Dict[str, Any]]:
    payload = {
        "codigoTabelaReferencia": codigo_tabela_referencia,
        "codigoTipoVeiculo": codigo_tipo_veiculo,
    }
    data = post_json(session, ENDPOINTS["marcas"], payload)
    out = []
    for item in data:
        out.append({
            "codigo": str(item.get("Value") or item.get("value") or item.get("Codigo") or item.get("codigo")),
            "nome": normalize_text(item.get("Label") or item.get("label") or item.get("Nome") or item.get("nome")),
        })
    return [x for x in out if x["codigo"] not in ("None", "", "nan")]


def get_modelos(session: Session, codigo_tabela_referencia: int, codigo_tipo_veiculo: int, codigo_marca: str) -> List[Dict[str, Any]]:
    payload = {
        "codigoTabelaReferencia": codigo_tabela_referencia,
        "codigoTipoVeiculo": codigo_tipo_veiculo,
        "codigoMarca": codigo_marca,
    }
    data = post_json(session, ENDPOINTS["modelos"], payload)

    # a FIPE costuma retornar {"Modelos":[...], "Anos":[...]}
    modelos = data.get("Modelos") or data.get("modelos") or data.get("Models") or []
    out = []
    for item in modelos:
        out.append({
            "codigo": str(item.get("Value") or item.get("value") or item.get("Codigo") or item.get("codigo")),
            "nome": normalize_text(item.get("Label") or item.get("label") or item.get("Nome") or item.get("nome")),
        })
    return [x for x in out if x["codigo"] not in ("None", "", "nan")]


def get_anos_modelo(
    session: Session,
    codigo_tabela_referencia: int,
    codigo_tipo_veiculo: int,
    codigo_marca: str,
    codigo_modelo: str,
) -> List[Dict[str, Any]]:
    payload = {
        "codigoTabelaReferencia": codigo_tabela_referencia,
        "codigoTipoVeiculo": codigo_tipo_veiculo,
        "codigoMarca": codigo_marca,
        "codigoModelo": codigo_modelo,
    }
    data = post_json(session, ENDPOINTS["ano_modelo"], payload)
    out = []
    for item in data:
        codigo = str(item.get("Value") or item.get("value") or item.get("Codigo") or item.get("codigo"))
        nome = normalize_text(item.get("Label") or item.get("label") or item.get("Nome") or item.get("nome"))
        out.append({"codigo": codigo, "nome": nome})
    return [x for x in out if x["codigo"] not in ("None", "", "nan")]


def get_valor(
    session: Session,
    codigo_tabela_referencia: int,
    codigo_tipo_veiculo: int,
    codigo_marca: str,
    codigo_modelo: str,
    year_model_code: str,
) -> Dict[str, Any]:
    ano, fuel_code = parse_year_model_code(year_model_code)
    payload = {
        "codigoTabelaReferencia": codigo_tabela_referencia,
        "codigoTipoVeiculo": codigo_tipo_veiculo,
        "codigoMarca": codigo_marca,
        "codigoModelo": codigo_modelo,
        "anoModelo": ano,
        "codigoTipoCombustivel": fuel_code,
        "ano": year_model_code,
        "tipoConsulta": "tradicional",
    }
    return post_json(session, ENDPOINTS["valor"], payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scraper histórico da Tabela FIPE.")
    parser.add_argument("--start", required=True, help="Mês inicial no formato YYYY-MM")
    parser.add_argument("--end", required=True, help="Mês final no formato YYYY-MM")
    parser.add_argument("--vehicle", default="carros", choices=["carros", "motos", "caminhoes"])
    parser.add_argument("--brands", default="", help='Lista opcional de marcas separadas por vírgula. Ex: "Toyota,Fiat,Ford"')
    parser.add_argument("--out-dir", default="data/raw", help="Diretório de saída")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    csv_path = os.path.join(args.out_dir, f"fipe_historico_{args.vehicle}.csv")
    log_path = os.path.join(args.out_dir, f"fipe_historico_{args.vehicle}.log.txt")

    brand_filter = {normalize_text(x).lower() for x in args.brands.split(",") if normalize_text(x)}

    vehicle_type_code = VEHICLE_TYPES[args.vehicle]
    session = create_session()

    refs = get_references(session)
    refs = filter_references(refs, args.start, args.end)
    if not refs:
        raise SystemExit(f"Nenhuma referência encontrada no intervalo {args.start} até {args.end}")

    buffer_rows: List[FipeRow] = []
    total_errors = 0

    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n=== INÍCIO {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        logf.write(f"intervalo={args.start}..{args.end} vehicle={args.vehicle}\n")

        for ref in refs:
            print(f"[REF] {ref.ym} (codigo={ref.codigo_tabela_referencia})")
            logf.write(f"[REF] {ref.ym} (codigo={ref.codigo_tabela_referencia})\n")

            try:
                marcas = get_marcas(session, ref.codigo_tabela_referencia, vehicle_type_code)
            except Exception as e:
                total_errors += 1
                logf.write(f"ERRO marcas ref={ref.codigo_tabela_referencia}: {e}\n")
                continue

            if brand_filter:
                marcas = [m for m in marcas if normalize_text(m["nome"]).lower() in brand_filter]

            for marca in marcas:
                print(f"  [MARCA] {marca['nome']} ({marca['codigo']})")
                try:
                    modelos = get_modelos(
                        session,
                        ref.codigo_tabela_referencia,
                        vehicle_type_code,
                        marca["codigo"],
                    )
                except Exception as e:
                    total_errors += 1
                    logf.write(
                        f"ERRO modelos ref={ref.codigo_tabela_referencia} marca={marca['codigo']}: {e}\n"
                    )
                    continue

                for modelo in modelos:
                    try:
                        anos_modelo = get_anos_modelo(
                            session,
                            ref.codigo_tabela_referencia,
                            vehicle_type_code,
                            marca["codigo"],
                            modelo["codigo"],
                        )
                    except Exception as e:
                        total_errors += 1
                        logf.write(
                            f"ERRO anos ref={ref.codigo_tabela_referencia} marca={marca['codigo']} "
                            f"modelo={modelo['codigo']}: {e}\n"
                        )
                        continue

                    for ano_item in anos_modelo:
                        ym_code = ano_item["codigo"]

                        try:
                            payload_resp = get_valor(
                                session,
                                ref.codigo_tabela_referencia,
                                vehicle_type_code,
                                marca["codigo"],
                                modelo["codigo"],
                                ym_code,
                            )

                            ano, fuel_code = parse_year_model_code(ym_code)

                            row = FipeRow(
                                reference_code=ref.codigo_tabela_referencia,
                                reference_month=normalize_text(ref.mes_raw),
                                reference_date=ref.date_str,
                                vehicle_type=vehicle_type_code,
                                brand_code=str(marca["codigo"]),
                                brand_name=normalize_text(marca["nome"]),
                                model_code=str(modelo["codigo"]),
                                model_name=normalize_text(payload_resp.get("Modelo") or payload_resp.get("modelo") or modelo["nome"]),
                                year_model_code=ym_code,
                                ano=ano,
                                fuel_code=fuel_code,
                                fuel_name=normalize_text(payload_resp.get("Combustivel") or payload_resp.get("combustivel") or ano_item["nome"]),
                                codigo_fipe=normalize_text(payload_resp.get("CodigoFipe") or payload_resp.get("codigoFipe") or ""),
                                autenticacao=normalize_text(payload_resp.get("Autenticacao") or payload_resp.get("autenticacao") or ""),
                                preco_raw=normalize_text(payload_resp.get("Valor") or payload_resp.get("valor") or ""),
                                preco_brl=parse_brl(payload_resp.get("Valor") or payload_resp.get("valor")),
                                mes_referencia_api=normalize_text(payload_resp.get("MesReferencia") or payload_resp.get("mesReferencia") or ""),
                                tipo_veiculo=int(payload_resp.get("TipoVeiculo") or payload_resp.get("tipoVeiculo") or vehicle_type_code),
                                sigla_combustivel=normalize_text(payload_resp.get("SiglaCombustivel") or payload_resp.get("siglaCombustivel") or ""),
                            )
                            buffer_rows.append(row)
                            total_written += 1

                        except Exception as e:
                            total_errors += 1
                            logf.write(
                                f"ERRO valor ref={ref.codigo_tabela_referencia} marca={marca['codigo']} "
                                f"modelo={modelo['codigo']} ano_modelo={ym_code}: {e}\n"
                            )

                            time.sleep(0.35)

                if buffer_rows:
                    append_rows_csv(csv_path, buffer_rows)
                    buffer_rows.clear()

        logf.write(f"FINAL total_written={total_written} total_errors={total_errors}\n")
        logf.write(f"=== FIM {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    print("\nConcluído.")
    print(f"CSV:     {csv_path}")
    print(f"Log: {log_path}")
    print(f"Linhas escritas: {total_written}")
    print(f"Erros: {total_errors}")


if __name__ == "__main__":
    main()