import os
from statistics import mean

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

APP_VERSION = os.getenv("C1NC0_APP_VERSION", "piloto-2026-10-03")

_client = None


def _supabase():
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SECRET_KEY", "").strip()
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL e SUPABASE_SECRET_KEY precisam estar configuradas."
            )
        _client = create_client(url, key)
    return _client


def registrar_analise(payload):
    dados = {
        "session_id": payload["session_id"],
        "origem": payload["origem"],
        "modo_modelo": payload["modo_modelo"],
        "modelos_concordaram": payload.get("modelos_concordaram"),
        "versao_app": APP_VERSION,
    }
    resposta = _supabase().table("c1nc0_analises").insert(dados).execute()
    return resposta.data[0] if resposta.data else dados


def registrar_feedback(payload):
    dados = {
        "session_id": payload["session_id"],
        "analise_id": payload.get("analise_id"),
        "origem": payload["origem"],
        "modo_modelo": payload["modo_modelo"],
        "modelos_concordaram": payload.get("modelos_concordaram"),
        "ajudou_analise": payload.get("ajudou_analise"),
        "buscaria_outras_fontes": payload.get("buscaria_outras_fontes"),
        "clareza_limites": payload.get("clareza_limites"),
        "clareza_comparacao_modelos": payload.get("clareza_comparacao_modelos"),
        "mudou_avaliacao": payload.get("mudou_avaliacao"),
        "parte_mais_util": payload.get("parte_mais_util"),
        "usaria_novamente": payload.get("usaria_novamente"),
        "comentario": (payload.get("comentario") or "").strip()[:2000] or None,
        "versao_app": APP_VERSION,
    }
    resposta = _supabase().table("c1nc0_feedback").insert(dados).execute()
    return resposta.data[0] if resposta.data else dados


def _media(rows, campo):
    valores = [r.get(campo) for r in rows if isinstance(r.get(campo), (int, float))]
    return round(mean(valores), 2) if valores else None


def _pct(rows, campo, valor):
    validos = [r.get(campo) for r in rows if r.get(campo) is not None]
    if not validos:
        return None
    return round(sum(v == valor for v in validos) / len(validos) * 100, 1)


def obter_resultados_piloto():
    analises = (
        _supabase()
        .table("c1nc0_analises")
        .select("*")
        .eq("versao_app", APP_VERSION)
        .execute()
        .data
        or []
    )
    feedbacks = (
        _supabase()
        .table("c1nc0_feedback")
        .select("*")
        .eq("versao_app", APP_VERSION)
        .execute()
        .data
        or []
    )

    sessoes = {r.get("session_id") for r in analises if r.get("session_id")}
    comparacoes = [r for r in analises if r.get("modo_modelo") == "comparar"]
    concordantes = [r for r in comparacoes if r.get("modelos_concordaram") is True]
    divergentes = [r for r in comparacoes if r.get("modelos_concordaram") is False]

    origem_ext = sum(r.get("origem") == "extensao" for r in analises)
    origem_site = sum(r.get("origem") == "site" for r in analises)

    modos = {
        "comparar": sum(r.get("modo_modelo") == "comparar" for r in analises),
        "gaussian": sum(r.get("modo_modelo") == "gaussian" for r in analises),
        "multinomial": sum(r.get("modo_modelo") == "multinomial" for r in analises),
    }

    return {
        "versao_app": APP_VERSION,
        "sessoes": len(sessoes),
        "analises": len(analises),
        "feedbacks": len(feedbacks),
        "taxa_feedback": round(len(feedbacks) / len(analises) * 100, 1) if analises else 0,
        "origem": {"extensao": origem_ext, "site": origem_site},
        "modos": modos,
        "comparacoes": {
            "total": len(comparacoes),
            "concordantes": len(concordantes),
            "divergentes": len(divergentes),
        },
        "guiding": {
            "ajudou_analise_media": _media(feedbacks, "ajudou_analise"),
            "clareza_limites_media": _media(feedbacks, "clareza_limites"),
            "clareza_comparacao_media": _media(
                feedbacks, "clareza_comparacao_modelos"
            ),
            "buscaria_outras_fontes_sim_pct": _pct(
                feedbacks, "buscaria_outras_fontes", "sim"
            ),
            "usaria_novamente_sim_pct": _pct(
                feedbacks, "usaria_novamente", "sim"
            ),
        },
    }
