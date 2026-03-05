
import asyncio
import json
import logging
import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple

import difflib

import requests
from requests import HTTPError, RequestException
from livekit.agents import RunContext, function_tool


RAVVIOINSIGHTS_BACKEND_URL = os.getenv("RAVVIOINSIGHTS_BACKEND_URL")
FALLBACK_BEARER_TOKEN = os.getenv("RAVVIOINSIGHTS_BACKEND_BEARER_TOKEN")
FALLBACK_TRANSCRIPT_ID = os.getenv("RAVVIOINSIGHTS_BACKEND_TRANSCRIPT_ID")

_VOICE_BANNED_CHARS_RE = re.compile(r"[*_~`#<>{}\\[\\]()%@|^]")


def _voice(text: str) -> str:
    if not text:
        return ""
    cleaned = _VOICE_BANNED_CHARS_RE.sub("", str(text))
    # Normalize spaces while preserving newlines (useful for voice progress blocks).
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in cleaned.split("\n")]
    normalized: List[str] = []
    last_blank = False
    for line in lines:
        if not line:
            if not last_blank:
                normalized.append("")
            last_blank = True
            continue
        normalized.append(line)
        last_blank = False
    while normalized and not normalized[0]:
        normalized.pop(0)
    while normalized and not normalized[-1]:
        normalized.pop()
    return "\n".join(normalized)


def _preferred_language(context: RunContext) -> str:
    try:
        userdata = context.userdata  # type: ignore[assignment]
    except ValueError:
        userdata = {}
    if isinstance(userdata, dict):
        lang = str(userdata.get("preferred_language") or "").strip().lower()
        if lang in {"en", "hi"}:
            return lang
    return "hi"


def _say(context: RunContext, *, en: str, hi: str) -> str:
    return _voice(hi if _preferred_language(context) == "hi" else en)


def _contains_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in (text or ""))


def _extract_first_number(text: str) -> Optional[str]:
    match = re.search(r"(-?\\d+(?:[\\.,]\\d+)?)", text or "")
    if not match:
        return None
    return match.group(1)


def _summarize_text(text: Optional[str], max_sentences: int = 2) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    summary = " ".join(sentences[:max_sentences]).strip()
    return summary or cleaned


def _parse_optional_json(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    candidate = value.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logging.warning("Ignoring malformed JSON payload: %s", candidate[:120])
        return None


def _format_graph_brief(graph: Dict[str, Any]) -> str:
    title = (graph.get("title") or graph.get("description") or graph.get("graph_id") or "Untitled graph").strip()
    active = "active" if graph.get("active", True) else "inactive"
    if not title:
        title = "Untitled graph"
    return _voice(f"{title}, {active}")


def _format_dashboard_scope_message(payload: Dict[str, Any], *, lang: str = "en") -> str:
    """
    Compose a concise status line from a dashboard scope update payload.
    """
    message = (payload.get("message") or payload.get("dashboard_result") or "").strip()
    action = (payload.get("action") or payload.get("dashboard_action") or "").strip()
    updated = payload.get("updated")
    graphs = payload.get("graphs") or payload.get("graphs_used") or []
    if isinstance(graphs, dict):
        graphs = list(graphs.values())
    if not message:
        message = "डैशबोर्ड अनुरोध पूरा हो गया।" if lang == "hi" else "Dashboard request processed."
    details: List[str] = []
    if isinstance(updated, int) and updated >= 0:
        if lang == "hi":
            details.append(f"{updated} आइटम प्रभावित हुए")
        else:
            details.append(f"{updated} item{'s' if updated != 1 else ''} affected")
    if isinstance(graphs, list) and graphs:
        preview = ", ".join(str(name) for name in graphs[:4] if name)
        if preview:
            if len(graphs) > 4:
                preview += ", …"
            details.append(f"लक्ष्य: {preview}" if lang == "hi" else f"Targets: {preview}")
    if action:
        if lang == "hi":
            action_map = {
                "activate": "सक्रिय",
                "deactivate": "निष्क्रिय",
                "exclusive": "एकमात्र",
            }
            action_text = action_map.get(action.lower(), action)
            message = f"{action_text} {message}"
        else:
            message = f"{action.capitalize()} {message}"
    if details:
        message = f"{message} {'; '.join(details)}"
    if lang == "hi" and not _contains_devanagari(message):
        return _voice("डैशबोर्ड अपडेट कर दिया है।")
    return _voice(message)


def _set_userdata_fields(context: RunContext, updates: Dict[str, Any]) -> None:
    try:
        userdata = context.userdata  # type: ignore[assignment]
    except ValueError:
        userdata = {}
    if not isinstance(userdata, dict):
        userdata = {}
    userdata.update({k: v for k, v in updates.items() if v is not None})
    # Avoid re-assigning userdata so other components holding a reference
    # (like language routing TTS) observe updates immediately.


def _wants_dashboard_pin(query: str) -> bool:
    lowered = (query or "").lower()
    if "dashboard" not in lowered:
        return False
    return any(
        phrase in lowered
        for phrase in (
            "add on dashboard",
            "add to dashboard",
            "pin to dashboard",
            "pin on dashboard",
            "save to dashboard",
            "save on dashboard",
        )
    )


def _extract_dashboard_target(query: str) -> Optional[str]:
    lowered = (query or "").lower()
    match = re.search(r"(?:to|on)\\s+(?:the\\s+)?(.+?)\\s+dashboard\\b", lowered)
    if not match:
        return None
    target = match.group(1).strip()
    if not target or target in {"a", "any", "new", "this", "that", "my"}:
        return None
    if len(target) > 60:
        target = target[:60].strip()
    return target or None


def _resolve_backend_context(context: RunContext) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    backend_url = RAVVIOINSIGHTS_BACKEND_URL or os.getenv("RAVVIOINSIGHTS_BACKEND_URL")
    if not backend_url:
        logging.error("Backend URL not configured for LiveKit agent tools.")
        return None, _say(context, en="Backend is not configured.", hi="बैकएंड कॉन्फ़िगर नहीं है।")

    try:
        userdata = context.userdata  # type: ignore[assignment]
    except ValueError:
        userdata = {}
    if not isinstance(userdata, dict):
        userdata = {}

    session_metadata = userdata.get("metadata")
    if not isinstance(session_metadata, dict):
        session_metadata = {}

    auth_token = userdata.get("auth_token") or session_metadata.get("auth_token") or FALLBACK_BEARER_TOKEN
    if not auth_token:
        logging.error("Missing authorization token for backend access.")
        return None, _say(context, en="Authorization is unavailable for the backend.", hi="बैकएंड के लिए ऑथराइज़ेशन उपलब्ध नहीं है।")

    transcript_id = (
        userdata.get("transcript_id")
        or session_metadata.get("transcript_id")
        or FALLBACK_TRANSCRIPT_ID
    )

    # Allow the active dashboard to be overridden per session via userdata.
    dashboard_id = (
        userdata.get("dashboard_id")
        or session_metadata.get("dashboard_id")
    )

    context_payload = {
        "backend_url": backend_url.rstrip("/"),
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}",
        },
        "transcript_id": transcript_id,
        "dashboard_id": dashboard_id,
        "session_metadata": session_metadata,
        "userdata": userdata,
    }
    return context_payload, None


async def _call_backend(
    context: RunContext,
    method: str,
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 15.0,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    if config is None:
        config, error = _resolve_backend_context(context)
    else:
        error = None
    if error or not config:
        return None, error or _say(context, en="Backend context unavailable.", hi="बैकएंड संदर्भ उपलब्ध नहीं है।")

    url = f"{config['backend_url']}{path}"
    request_kwargs: Dict[str, Any] = {
        "headers": config["headers"],
        "timeout": timeout,
    }
    if params:
        request_kwargs["params"] = params
    method_upper = method.upper()
    if payload is not None and method_upper not in {"GET", "DELETE"}:
        request_kwargs["json"] = payload
    elif payload is not None and method_upper == "DELETE":
        request_kwargs["json"] = payload

    try:
        response = await asyncio.to_thread(
            requests.request,
            method_upper,
            url,
            **request_kwargs,
        )
    except RequestException as exc:
        logging.error("Request to %s %s failed before response: %s", method_upper, path, exc)
        return None, _say(
            context,
            en="Backend request failed; please try again later.",
            hi="बैकएंड अनुरोध विफल रहा, कृपया थोड़ी देर बाद फिर कोशिश करें।",
        )

    try:
        response.raise_for_status()
    except HTTPError as exc:
        detail = ""
        if exc.response is not None:
            detail = exc.response.text.strip()
        logging.error("Backend responded with error for %s %s: %s", method_upper, path, detail or exc)
        if detail:
            return None, _say(context, en=f"Backend error: {detail}", hi=f"बैकएंड त्रुटि: {detail}")
        return None, _say(context, en="Backend returned an error.", hi="बैकएंड से त्रुटि आई है।")

    if not response.content:
        return {}, None
    try:
        return response.json(), None
    except ValueError:
        return response.text.strip(), None


@function_tool()
async def check_health_status(
    context: RunContext,  # type: ignore
    verbose: bool = True,
) -> str:
    """
    Check the local health API at http://localhost:8000/health.

    Use this tool whenever the user asks about system health, diagnostics, uptime,
    or anything related to the service being healthy. Returns the raw response body
    if the endpoint responds with HTTP 200.
    """
    url = "http://localhost:8000/health"

    try:
        response = await asyncio.to_thread(requests.get, url, timeout=5)
        response.raise_for_status()
        body = response.text.strip()
        if not body:
            body = "Health endpoint returned an empty body but status was 200."
        if verbose:
            logging.info("Health check successful: %s", body)
            return _say(context, en=body, hi=body)
        logging.info("Health check successful")
        return _say(context, en="Health check OK.", hi="सिस्टम ठीक है।")
    except Exception as exc:
        logging.error("Health check failed: %s", exc)
        return _say(context, en="Health check failed; please check the service manually.", hi="हेल्थ चेक असफल रहा, कृपया सिस्टम की जाँच करें।")


@function_tool()
async def voice_set_language(
    context: RunContext,  # type: ignore
    language: str,
) -> str:
    """
    Set the preferred conversation language for this voice session.

    Use this when the user asks to "talk in Hindi" or "talk in English".
    This updates session userdata and (when available) updates the session STT language.
    """
    raw = (language or "").strip().lower()
    if not raw:
        return _say(context, en="Please specify English or Hindi.", hi="कृपया अंग्रेज़ी या हिंदी बताइए।")

    if raw in {"hi", "hindi", "hin", "हिंदी"}:
        preferred = "hi"
    elif raw in {"en", "english", "eng"}:
        preferred = "en"
    else:
        return _say(context, en="I can switch between English and Hindi only.", hi="मैं सिर्फ़ अंग्रेज़ी और हिंदी के बीच स्विच कर सकता हूँ।")

    stt_lang = os.getenv("DEEPGRAM_STT_LANGUAGE_HI", "hi") if preferred == "hi" else os.getenv("DEEPGRAM_STT_LANGUAGE_EN", "en-IN")
    _set_userdata_fields(
        context,
        {
            "preferred_language": preferred,
            "stt_language": stt_lang,
            "pending_language_switch": None,
        },
    )

    stt_impl = getattr(getattr(context, "session", None), "stt", None)
    if stt_impl is not None and hasattr(stt_impl, "update_options"):
        try:
            stt_impl.update_options(language=stt_lang)
        except Exception as exc:
            logging.warning("Failed to update STT language to %s: %s", stt_lang, exc)

    if preferred == "hi":
        return _voice("समझ गया, अब हम हिंदी में संवाद करेंगे।")
    return _voice("Understood, we will continue in English.")


@function_tool()
async def voice_request_language_switch(
    context: RunContext,  # type: ignore
    language: str,
) -> str:
    """
    Ask the user to confirm before changing the conversation language.

    Use this when the user requests switching languages; do not change language yet.
    """
    raw = (language or "").strip().lower()
    if raw in {"hi", "hindi", "hin", "हिंदी"}:
        target = "hi"
    elif raw in {"en", "english", "eng"}:
        target = "en"
    else:
        return _say(context, en="I can switch between English and Hindi only.", hi="मैं सिर्फ़ अंग्रेज़ी और हिंदी के बीच स्विच कर सकता हूँ।")

    current = _preferred_language(context)
    if target == current:
        if target == "hi":
            return _voice("हम पहले से हिंदी में बात कर रहे हैं।")
        return _say(context, en="We are already speaking in English.", hi="हम पहले से अंग्रेज़ी में बात कर रहे हैं।")

    _set_userdata_fields(context, {"pending_language_switch": target})
    if target == "hi":
        # Ask in the current language, but phrase it as a direct question.
        return _say(context, en="So, can we switch to Hindi now?", hi="तो क्या हम अभी हिंदी में स्विच करें?")
    return _say(context, en="So, can we switch to English now?", hi="तो क्या हम अभी अंग्रेज़ी में स्विच करें?")


@function_tool()
async def voice_confirm_language_switch(
    context: RunContext,  # type: ignore
    confirm: bool = True,
) -> str:
    """
    Confirm or cancel a pending language switch requested via voice_request_language_switch.
    """
    try:
        userdata = context.userdata  # type: ignore[assignment]
    except ValueError:
        userdata = {}
    userdata = userdata if isinstance(userdata, dict) else {}
    pending = str(userdata.get("pending_language_switch") or "").strip().lower()
    if pending not in {"en", "hi"}:
        return _say(context, en="There is no pending language change.", hi="कोई लंबित भाषा बदलाव नहीं है।")

    if not confirm:
        _set_userdata_fields(context, {"pending_language_switch": None})
        return _say(context, en="Okay, staying in the current language.", hi="ठीक है, हम वर्तमान भाषा में ही रहेंगे।")

    # Perform the switch now.
    return await voice_set_language(context, pending)


@function_tool()
async def process_user_query(
    context: RunContext,  # type: ignore
    query: str,
    title: Optional[str] = None,
    metadata: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> str:
    """
    Send a business or analytics question to the Process Query API to obtain SQL and metadata.

    Use this when the user asks for business insights, analytics, or database-backed answers.
    The optional `metadata` parameter can include serialized JSON with extra context.
    """
    if not query:
        return _say(context, en="Cannot process an empty query.", hi="खाली सवाल प्रोसेस नहीं कर सकता हूँ।")

    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(
            context,
            en=(error or "Backend configuration is unavailable."),
            hi="बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।",
        )

    transcript_id = config.get("transcript_id")
    if not transcript_id:
        logging.error("process_user_query missing transcript id in session metadata")
        return _say(context, en="Process Query API call failed; transcript context is unavailable.", hi="ट्रांसक्रिप्ट संदर्भ उपलब्ध नहीं है, इसलिए अनुरोध पूरा नहीं हो पाया।")

    generated_title = title or "Sales Query"
    request_metadata: Dict[str, Any] = {}
    parsed_metadata = _parse_optional_json(metadata)
    if isinstance(parsed_metadata, dict):
        request_metadata = dict(parsed_metadata)
    if "source" not in request_metadata:
        request_metadata["source"] = "voice-agent"
    preferred_lang = _preferred_language(context)
    request_metadata.setdefault("preferred_language", preferred_lang)
    request_metadata.setdefault("response_language", preferred_lang)

    payload = {
        "natural_language_query": query,
        "transcript_id": transcript_id,
        "title": generated_title or "User query",
        "metadata": request_metadata,
        "conversation_context": (
            (conversation_context or "User asked a business query.")
            + (
                " response_language=hi."
                if preferred_lang == "hi"
                else " response_language=en."
            )
        ).strip(),
    }

    data, error = await _call_backend(context, "POST", "/process_query", payload=payload, timeout=20.0, config=config)
    if error:
        return _say(context, en=error, hi="अनुरोध पूरा नहीं हो पाया।")
    if not isinstance(data, dict):
        logging.info("Process Query API success with non-dict response")
        if preferred_lang == "hi":
            return _voice("परिणाम तैयार है।")
        return _voice(data if isinstance(data, str) else json.dumps(data))

    description_text: Optional[str] = None
    supplemental_hint: Optional[str] = None

    transcript_id = data.get("transcript_id")
    if transcript_id:
        try:
            userdata = context.userdata  # type: ignore[assignment]
        except ValueError:
            userdata = {}
        if isinstance(userdata, dict):
            if userdata.get("transcript_id") != transcript_id:
                userdata["transcript_id"] = transcript_id
                try:
                    context.userdata = userdata  # type: ignore[attr-defined]
                except Exception:
                    pass
    chat_id_assistant = data.get("chat_id_assistant")
    chat_id_user = data.get("chat_id_user")
    if transcript_id and (chat_id_assistant or chat_id_user):
        _set_userdata_fields(
            context,
            {
                "transcript_id": transcript_id,
                "last_chat_id_assistant": chat_id_assistant,
                "last_chat_id_user": chat_id_user,
                "last_user_query": query,
            },
        )
    status = data.get("status")
    message = (data.get("message") or "").strip()
    if transcript_id and chat_id_assistant:
        description_data, desc_err = await _call_backend(
            context,
            "GET",
            f"/get_description/{transcript_id}/{chat_id_assistant}",
            timeout=10.0,
            config=config,
        )
        if not desc_err and isinstance(description_data, dict):
            description_text = _summarize_text(description_data.get("description"))
        elif desc_err:
            logging.warning("Failed to fetch description for transcript %s chat %s: %s", transcript_id, chat_id_assistant, desc_err)

        tables_data, tables_err = await _call_backend(
            context,
            "GET",
            f"/get_tables/{transcript_id}/{chat_id_assistant}",
            timeout=10.0,
            config=config,
        )
        if not tables_err and isinstance(tables_data, dict):
            record_count = tables_data.get("record_count")
            if isinstance(record_count, int):
                if preferred_lang == "hi":
                    supplemental_hint = f"{record_count} रिकॉर्ड मिले।"
                else:
                    supplemental_hint = f"{record_count} record{'s' if record_count != 1 else ''} returned."
        elif tables_err:
            logging.debug("Failed to fetch tables metadata: %s", tables_err)

    summary_parts: List[str] = []
    if description_text:
        summary_parts.append(description_text)
    if supplemental_hint:
        summary_parts.append(supplemental_hint)
    if not summary_parts and message:
        summary_parts.append(message)
    if summary_parts:
        final_summary = " ".join(summary_parts)
    else:
        final_summary = "आपका अनुरोध सफलतापूर्वक प्रोसेस हो गया है।" if preferred_lang == "hi" else "Query processed successfully."
    if len(final_summary) > 400:
        final_summary = final_summary[:397].rstrip() + "..."
    if status and status not in {"success", ""}:
        if preferred_lang == "hi":
            final_summary = f"स्थिति {status}: {final_summary}"
        else:
            final_summary = f"{status.capitalize()}: {final_summary}"

    if transcript_id and chat_id_assistant and _wants_dashboard_pin(query):
        try:
            userdata = context.userdata  # type: ignore[assignment]
        except ValueError:
            userdata = {}
        userdata = userdata if isinstance(userdata, dict) else {}
        if userdata.get("last_pinned_chat_id_assistant") != chat_id_assistant:
            target_dashboard = _extract_dashboard_target(query)
            pin_result = await dashboard_add_latest_graph_to_dashboard(
                context,
                dashboard_name_or_id=target_dashboard,
                max_graphs=1,
            )
            pin_text = (pin_result or "").strip().lower()
            pin_succeeded = pin_text.startswith("added") or ("ग्राफ" in pin_text and ("जोड़" in pin_text or "पिन" in pin_text))
            if pin_succeeded:
                _set_userdata_fields(context, {"last_pinned_chat_id_assistant": chat_id_assistant})
                try:
                    userdata = context.userdata  # type: ignore[assignment]
                except ValueError:
                    userdata = {}
                dash_name = ""
                if isinstance(userdata, dict):
                    dash_name = str(userdata.get("dashboard_name") or "").strip()
                if preferred_lang == "hi":
                    if dash_name:
                        final_summary = f"{final_summary.rstrip('।.')} और नवीनतम ग्राफ {dash_name} डैशबोर्ड पर पिन कर दिया है।"
                    else:
                        final_summary = f"{final_summary.rstrip('।.')} और नवीनतम ग्राफ डैशबोर्ड पर पिन कर दिया है।"
                else:
                    if dash_name:
                        final_summary = f"{final_summary.rstrip('.')} and pinned the latest graph to {dash_name}."
                    else:
                        final_summary = f"{final_summary.rstrip('.')} and pinned the latest graph to a dashboard."
    logging.info("Process Query API success")

    try:
        userdata = context.userdata  # type: ignore[assignment]
    except ValueError:
        userdata = {}
    voice_progress_enabled = bool(userdata.get("voice_progress_enabled")) if isinstance(userdata, dict) else False

    if preferred_lang == "hi" and not _contains_devanagari(final_summary):
        number = _extract_first_number(final_summary)
        lower_query = (query or "").lower()
        if ("employee" in lower_query) or ("employees" in lower_query) or _contains_devanagari(query):
            if "कर्मचारी" in (query or "") or "employee" in lower_query or "employees" in lower_query:
                if number:
                    return _voice(f"कुल {number} कर्मचारी हैं।")
                return _voice("कुल कर्मचारियों की संख्या निकाल ली है।")
        return _voice("परिणाम तैयार है।")

    if voice_progress_enabled:
        if preferred_lang == "hi":
            progress_block = (
                "आपकी रिक्वेस्ट प्रोसेस कर रहा हूँ\n\n"
                "आपके प्रश्न का विश्लेषण कर रहा हूँ\n"
                "क्वेरी बना रहा हूँ\n"
                "परिणाम का सार तैयार कर रहा हूँ\n"
                "परिणामों का विज़ुअलाइज़ेशन बना रहा हूँ\n"
                "अंतिम उत्तर तैयार कर रहा हूँ"
            )
        else:
            progress_block = (
                "Processing your request\n\n"
                "Analysing your Query\n"
                "Building Query\n"
                "Summarizing Result\n"
                "Visualizing Results\n"
                "Finalizing Answer"
            )
        combined = f"{progress_block}\n\n{final_summary}".strip()
        return _voice(combined)

    return _voice(final_summary)


@function_tool()
async def dashboard_list_dashboards(
    context: RunContext,  # type: ignore
    name_contains: Optional[str] = None,
    include_ids: bool = False,
) -> str:
    """
    List available dashboards for the current user.

    Use this when the user asks which dashboards exist or wants to choose between them.
    """
    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

    data, error = await _call_backend(
        context,
        "GET",
        "/dashboards",
        config=config,
    )
    if error:
        return _voice(error)
    if not isinstance(data, dict):
        return _say(context, en="Unable to read dashboards.", hi="डैशबोर्ड पढ़ नहीं पा रहा हूँ।")

    dashboards = data.get("dashboards") or []
    if not dashboards:
        return _say(context, en="You do not have any dashboards yet.", hi="आपके पास अभी कोई डैशबोर्ड नहीं है।")

    if name_contains:
        needle = name_contains.strip().lower()
        if needle:
            dashboards = [
                dash
                for dash in dashboards
                if needle in str(dash.get("name") or "").lower()
            ]
            if not dashboards:
                return _say(context, en=f"No dashboards match '{name_contains.strip()}'.", hi=f"'{name_contains.strip()}' नाम का कोई डैशबोर्ड नहीं मिला।")

    # Persist a simple index of dashboards in userdata for later resolution.
    try:
        userdata = context.userdata  # type: ignore[assignment]
    except ValueError:
        userdata = {}
    if isinstance(userdata, dict):
        index: Dict[str, str] = {}
        for dash in dashboards:
            dash_id = str(dash.get("dashboard_id") or "").strip()
            name = str(dash.get("name") or "").strip()
            if not dash_id:
                continue
            key_id = dash_id.lower()
            index[key_id] = dash_id
            if name:
                key_name = re.sub(r"[^a-z0-9]+", "", name.lower())
                if key_name:
                    index[key_name] = dash_id
        userdata["dashboards_index"] = index
        try:
            context.userdata = userdata  # type: ignore[attr-defined]
        except Exception:
            pass

    # Return dashboard titles only, per voice output requirements.
    titles: List[str] = []
    default_untitled = "अनाम डैशबोर्ड" if _preferred_language(context) == "hi" else "Untitled dashboard"
    for dash in dashboards:
        name = (dash.get("name") or default_untitled).strip()
        if name:
            if include_ids:
                dash_id = (dash.get("dashboard_id") or "").strip()
                titles.append(_voice(f"{name}, id {dash_id}") if dash_id else _voice(name))
            else:
                titles.append(name)
    if not titles:
        return _say(context, en="You do not have any named dashboards yet.", hi="आपके पास अभी कोई नाम वाला डैशबोर्ड नहीं है।")
    listing = ", ".join(titles)
    return _say(context, en=f"Your dashboards are: {listing}.", hi=f"आपके डैशबोर्ड हैं: {listing}।")


@function_tool()
async def dashboard_create_dashboard(
    context: RunContext,  # type: ignore
    reason: Optional[str] = None,
) -> str:
    """
    Create a new dashboard and make it the active dashboard for subsequent actions.

    Use this when the user asks to create a dashboard or says "add on dashboard" without naming a dashboard.
    """
    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

    data, error = await _call_backend(
        context,
        "POST",
        "/dashboard/new",
        payload={},
        timeout=15.0,
        config=config,
    )
    if error:
        return _voice(error)
    if not isinstance(data, dict):
        return _say(context, en="Created a new dashboard.", hi="एक नया डैशबोर्ड बना दिया है।")
    dashboard = data.get("dashboard") if isinstance(data.get("dashboard"), dict) else {}
    dash_id = str(dashboard.get("dashboard_id") or "").strip()
    default_name = "नया डैशबोर्ड" if _preferred_language(context) == "hi" else "New dashboard"
    dash_name = str(dashboard.get("name") or default_name).strip()
    if dash_id:
        _set_userdata_fields(context, {"dashboard_id": dash_id, "dashboard_name": dash_name})
    return _say(context, en=f"Created and selected dashboard {dash_name}.", hi=f"डैशबोर्ड {dash_name} बना कर चुन लिया है।")


@function_tool()
async def dashboard_set_active_dashboard(
    context: RunContext,  # type: ignore
    dashboard_name_or_id: str,
) -> str:
    """
    Select which dashboard should be considered active for subsequent dashboard tools.

    Use this when the user refers to a specific dashboard by name or id (for example 'Sales dashboard').
    """
    query = (dashboard_name_or_id or "").strip()
    if not query:
        return _say(context, en="Please provide a dashboard name or id to select.", hi="कृपया डैशबोर्ड का नाम या आईडी बताइए।")

    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

    data, error = await _call_backend(
        context,
        "GET",
        "/dashboards",
        config=config,
    )
    if error:
        return _voice(error)
    if not isinstance(data, dict):
        return _say(context, en="Unable to read dashboards.", hi="डैशबोर्ड पढ़ नहीं पा रहा हूँ।")

    dashboards = data.get("dashboards") or []
    if not dashboards:
        return _say(context, en="You do not have any dashboards yet.", hi="आपके पास अभी कोई डैशबोर्ड नहीं है।")

    lower_query = query.lower()
    normalized_query = re.sub(r"[^a-z0-9]+", "", lower_query)

    def _match_score(dash: Dict[str, Any]) -> int:
        name = (dash.get("name") or "").strip()
        dash_id = (dash.get("dashboard_id") or "").strip()
        if not name and not dash_id:
            return 0
        if dash_id == query:
            return 100
        if dash_id.lower() == lower_query:
            return 90
        name_lower = name.lower()
        if name_lower == lower_query:
            return 80
        norm_name = re.sub(r"[^a-z0-9]+", "", name_lower)
        score = 0
        if norm_name == normalized_query and norm_name:
            score = 75
        elif normalized_query and norm_name.startswith(normalized_query):
            score = 60
        elif normalized_query and normalized_query in norm_name:
            score = 50
        return score

    best_dash = None
    best_score = 0
    for dash in dashboards:
        score = _match_score(dash)
        if score > best_score:
            best_score = score
            best_dash = dash

    if not best_dash:
        return _say(context, en=f"No dashboard matched {query}.", hi=f"{query} नाम का कोई डैशबोर्ड नहीं मिला।")

    dash_id = (best_dash.get("dashboard_id") or "").strip()
    dash_name = (best_dash.get("name") or "Untitled dashboard").strip()
    if not dash_id:
        return _say(context, en=f"Unable to identify a dashboard for {query}.", hi=f"{query} के लिए डैशबोर्ड पहचान नहीं पा रहा हूँ।")

    _set_userdata_fields(context, {"dashboard_id": dash_id, "dashboard_name": dash_name})
    return _say(context, en=f"Using dashboard {dash_name} for dashboard actions.", hi=f"अब {dash_name} डैशबोर्ड इस्तेमाल कर रहा हूँ।")


@function_tool()
async def dashboard_list_graphs(
    context: RunContext,  # type: ignore
    active_only: bool = True,
) -> str:
    """
    Retrieve the registered dashboard graphs.

    Use this when a user asks which dashboard visuals are available or wants to confirm what is pinned.
    """
    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

    dashboard_id = config.get("dashboard_id")
    if not dashboard_id:
        return _say(context, en="No dashboard is selected yet.", hi="अभी कोई डैशबोर्ड चुना नहीं गया है।")

    data, error = await _call_backend(context, "GET", f"/dashboard/{dashboard_id}", config=config)
    if error:
        return _voice(error)
    if not isinstance(data, dict):
        return _say(context, en="Unable to read the dashboard graphs.", hi="डैशबोर्ड के ग्राफ पढ़ नहीं पा रहा हूँ।")
    graphs = data.get("graphs") or []
    if not graphs:
        dashboard = data.get("dashboard") if isinstance(data.get("dashboard"), dict) else {}
        dash_name = (dashboard.get("name") or config.get("userdata", {}).get("dashboard_name") or "this").strip() if isinstance(dashboard, dict) else "this"
        return _say(context, en=f"No graphs are saved on {dash_name} dashboard.", hi=f"{dash_name} डैशबोर्ड पर अभी कोई ग्राफ सेव नहीं है।")

    dashboard = data.get("dashboard") if isinstance(data.get("dashboard"), dict) else {}
    dash_name = (dashboard.get("name") or "").strip() if isinstance(dashboard, dict) else ""
    if not dash_name:
        try:
            userdata = context.userdata  # type: ignore[assignment]
        except ValueError:
            userdata = {}
        if isinstance(userdata, dict):
            fallback_name = "चुना हुआ" if _preferred_language(context) == "hi" else "the selected"
            dash_name = str(userdata.get("dashboard_name") or fallback_name).strip()
        else:
            dash_name = "चुना हुआ" if _preferred_language(context) == "hi" else "the selected"

    titles: List[str] = []
    default_graph_title = "ग्राफ" if _preferred_language(context) == "hi" else "Graph"
    for graph in graphs[:8]:
        if not isinstance(graph, dict):
            continue
        title = (graph.get("summary") or graph.get("chart_type") or graph.get("graph_id") or default_graph_title).strip()
        if title:
            titles.append(title)
    if not titles:
        return _say(context, en=f"Graphs are saved on {dash_name} dashboard, but none have readable titles.", hi=f"{dash_name} डैशबोर्ड पर ग्राफ हैं, लेकिन उनके नाम स्पष्ट नहीं हैं।")
    listing = ", ".join(titles)
    if len(graphs) > 8:
        listing = f"{listing}, and {len(graphs) - 8} more"
    return _say(context, en=f"Graphs on {dash_name} dashboard are: {listing}.", hi=f"{dash_name} डैशबोर्ड पर ये ग्राफ हैं: {listing}।")


@function_tool()
async def dashboard_query_graphs(
    context: RunContext,  # type: ignore
    question: str,
) -> str:
    """
    Ask a natural language question or issue an instruction about the dashboard graphs.

    Use this for conversational dashboard updates, removals, and insight requests. The backend
    will parse the intent (add/remove/query) and respond accordingly.
    """
    if not question or not question.strip():
        return _say(context, en="Please provide a dashboard question to answer.", hi="कृपया डैशबोर्ड के बारे में सवाल बताइए।")
    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

    payload: Dict[str, Any] = {"question": question}
    dashboard_id = config.get("dashboard_id")
    if dashboard_id:
        payload["dashboard_id"] = dashboard_id

    data, error = await _call_backend(
        context,
        "POST",
        "/dashboard/graphs/query",
        payload=payload,
        timeout=20.0,
        config=config,
    )
    if error:
        return _voice(error)
    if not isinstance(data, dict):
        return _say(context, en=(data if isinstance(data, str) else "Dashboard query processed."), hi="डैशबोर्ड अनुरोध पूरा हो गया।")

    if data.get("type") == "scope_update":
        return _format_dashboard_scope_message(data, lang=_preferred_language(context))

    message = (data.get("message") or "Dashboard query processed.").strip()
    graphs_used = data.get("graphs_used") or []
    if graphs_used:
        preview = ", ".join(str(name) for name in graphs_used[:4] if name)
        if preview:
            if len(graphs_used) > 4:
                preview += ", …"
            message = f"{message} Graphs used: {preview}."
    updates = data.get("updated")
    if isinstance(updates, int) and updates > 0:
        message = f"{message} {updates} item{'s' if updates != 1 else ''} affected."
    # If Hindi is preferred, keep the tool output in Hindi so the spoken response stays Hindi.
    if _preferred_language(context) == "hi":
        return _voice("डैशबोर्ड अनुरोध पूरा हो गया।")
    return _voice(message)


@function_tool()
async def dashboard_add_latest_graph_to_dashboard(
    context: RunContext,  # type: ignore
    dashboard_name_or_id: Optional[str] = None,
    max_graphs: int = 1,
) -> str:
    """
    Create a dashboard if needed and pin the most recent query visualization(s) to it.

    Use this when the user says "add on dashboard", "pin this", or asks to add the last graph to a dashboard.
    """
    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

    try:
        userdata = context.userdata  # type: ignore[assignment]
    except ValueError:
        userdata = {}
    userdata = userdata if isinstance(userdata, dict) else {}

    transcript_id = config.get("transcript_id")
    chat_id_assistant = userdata.get("last_chat_id_assistant")
    if not transcript_id or not chat_id_assistant:
        return _say(
            context,
            en="I do not have a recent visualization to add, please ask a data question first.",
            hi="मेरे पास जोड़ने के लिए हाल का कोई ग्राफ नहीं है, पहले कोई डेटा सवाल पूछिए।",
        )

    sql_payload, sql_error = await _call_backend(
        context,
        "GET",
        f"/get_sql/{transcript_id}/{chat_id_assistant}",
        timeout=15.0,
        config=config,
    )
    sql_query = ""
    if isinstance(sql_payload, dict):
        sql_query = str(sql_payload.get("sql_query") or "").strip()
    if not sql_query:
        sql_query = "SELECT 1"

    graphs_payload, graphs_error = await _call_backend(
        context,
        "GET",
        f"/get_graph/{transcript_id}/{chat_id_assistant}",
        timeout=20.0,
        config=config,
    )
    if graphs_error:
        return _voice(graphs_error)
    if not isinstance(graphs_payload, dict):
        return _say(context, en="Unable to load the latest visualization payload.", hi="नवीनतम ग्राफ लोड नहीं हो पाया।")
    graphs = graphs_payload.get("graphs") or []
    if not isinstance(graphs, list) or not graphs:
        return _say(context, en="No visualization was found to add to a dashboard.", hi="डैशबोर्ड में जोड़ने के लिए कोई ग्राफ नहीं मिला।")

    target = (dashboard_name_or_id or "").strip()
    if target:
        await dashboard_set_active_dashboard(context, target)
        config, error = _resolve_backend_context(context)
        if error or not config:
            return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))
    else:
        created_msg = await dashboard_create_dashboard(context)
        config, error = _resolve_backend_context(context)
        if error or not config:
            return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))
        logging.info("dashboard_add_latest_graph_to_dashboard: %s", created_msg)

    dashboard_id = config.get("dashboard_id")
    if not dashboard_id:
        return _say(context, en="Unable to determine which dashboard to add the graph to.", hi="किस डैशबोर्ड में जोड़ना है, यह तय नहीं हो पाया।")

    dashboard_name = ""
    try:
        userdata = context.userdata  # type: ignore[assignment]
    except ValueError:
        userdata = {}
    if isinstance(userdata, dict):
        dashboard_name = str(userdata.get("dashboard_name") or "").strip()

    count = 0
    pinned_titles: List[str] = []
    default_pinned_title = "पिन किया हुआ ग्राफ" if _preferred_language(context) == "hi" else "Pinned graph"
    for graph in graphs[: max(1, int(max_graphs))]:
        if not isinstance(graph, dict):
            continue
        title = str(graph.get("title") or graph.get("insight") or graph.get("query") or default_pinned_title).strip()
        chart_type = str(graph.get("graph_type") or graph.get("type") or "").strip() or None

        # The dashboard UI expects `data_json` to contain:
        # - `data`: an array of records
        # - `description`: a string
        graph_data = graph.get("data")
        data_rows: List[Dict[str, Any]] = []
        if isinstance(graph_data, list):
            data_rows = [row for row in graph_data if isinstance(row, dict)]

        description = str(graph.get("insight") or "").strip()
        if not description and isinstance(graph.get("summary"), str):
            description = str(graph.get("summary") or "").strip()
        if not description:
            description = title

        data_json: Dict[str, Any] = {
            "data": data_rows,
            "description": description,
            "source": "process_query",
            "title": title,
            "chart_type": chart_type,
            "graph": graph,  # keep the full graph payload for future use/debugging
        }

        payload = {
            "sql": sql_query,
            "summary": (title or description)[:240] if (title or description) else None,
            "chart_type": chart_type,
            "data_json": data_json,
        }
        _, save_error = await _call_backend(
            context,
            "POST",
            f"/dashboard/{dashboard_id}/graph",
            payload=payload,
            timeout=20.0,
            config=config,
        )
        if save_error:
            logging.warning("Failed to pin graph to dashboard %s: %s", dashboard_id, save_error)
            continue
        count += 1
        if title:
            pinned_titles.append(title)

    if count == 0:
        return _say(context, en="I could not pin the latest graph to the dashboard.", hi="मैं नवीनतम ग्राफ को डैशबोर्ड पर पिन नहीं कर पाया।")
    if not dashboard_name:
        dashboard_name = "डैशबोर्ड" if _preferred_language(context) == "hi" else "the dashboard"
    if pinned_titles:
        preview = ", ".join(pinned_titles[:3])
        return _say(context, en=f"Added {count} graph to {dashboard_name}: {preview}.", hi=f"{dashboard_name} में {count} ग्राफ जोड़ दिया: {preview}।")
    return _say(context, en=f"Added {count} graph to {dashboard_name}.", hi=f"{dashboard_name} में {count} ग्राफ जोड़ दिया।")


@function_tool()
async def dashboard_register_graph(
    context: RunContext,  # type: ignore
    title: str,
    graph_type: Optional[str] = None,
    description: Optional[str] = None,
    metadata_json: Optional[str] = None,
    figure_json: Optional[str] = None,
    summary_json: Optional[str] = None,
    html_content: Optional[str] = None,
    graph_id: Optional[str] = None,
) -> str:
    """
    Register or update a dashboard graph with the backend.

    Use this when the user explicitly asks to pin a prepared visualization or update its metadata.
    """
    if not title or not title.strip():
        return _say(context, en="A title is required to register a dashboard graph.", hi="डैशबोर्ड ग्राफ रजिस्टर करने के लिए शीर्षक जरूरी है।")

    payload: Dict[str, Any] = {
        "title": title.strip(),
    }
    if graph_type:
        payload["graph_type"] = graph_type
    if description:
        payload["description"] = description
    if html_content:
        payload["html_content"] = html_content
    if graph_id:
        payload["graph_id"] = graph_id

    metadata = _parse_optional_json(metadata_json)
    if metadata is not None:
        payload["metadata"] = metadata
    figure = _parse_optional_json(figure_json)
    if figure is not None:
        payload["figure"] = figure
    summary = _parse_optional_json(summary_json)
    if summary is not None:
        payload["summary"] = summary

    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

    dashboard_id = config.get("dashboard_id")
    if dashboard_id:
        payload["dashboard_id"] = dashboard_id
        metadata = payload.setdefault("metadata", {}) or {}
        if isinstance(metadata, dict):
            metadata.setdefault("dashboard_id", dashboard_id)

    data, error = await _call_backend(
        context,
        "POST",
        "/dashboard/graphs",
        payload=payload,
        config=config,
    )
    if error:
        return _voice(error)
    if not isinstance(data, dict):
        return _say(context, en="Dashboard graph registered.", hi="डैशबोर्ड ग्राफ रजिस्टर कर दिया है।")
    graph = data.get("graph")
    if isinstance(graph, dict):
        identifier = graph.get("graph_id") or graph.get("id") or "new graph"
        status = "active" if graph.get("active", True) else "inactive"
        if _preferred_language(context) == "hi":
            return _voice(f"डैशबोर्ड ग्राफ रजिस्टर हो गया: {graph.get('title', title)}।")
        return _voice(f"Registered dashboard graph '{graph.get('title', title)}' ({status}, id={identifier}).")
    return _say(context, en="Dashboard graph registered.", hi="डैशबोर्ड ग्राफ रजिस्टर कर दिया है।")


@function_tool()
async def dashboard_remove_graph(
    context: RunContext,  # type: ignore
    graph_identifier: Optional[str] = None,
) -> str:
    """
    Remove a dashboard graph by id or title.

    Use this when the user asks to delete or unpin a specific dashboard visualization.
    Leaving the identifier blank will remove the most recent active graph.
    """
    query = (graph_identifier or "").strip()
    if not query:
        config, error = _resolve_backend_context(context)
        if error or not config:
            return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

        dashboard_id = config.get("dashboard_id")

        payload: Dict[str, Any] = {"question": "remove the latest graph from the dashboard"}
        if dashboard_id:
            payload["dashboard_id"] = dashboard_id

        data, error = await _call_backend(
            context,
            "POST",
            "/dashboard/graphs/query",
            payload=payload,
            timeout=15.0,
            config=config,
        )
        if error:
            return _say(context, en=error, hi="नवीनतम ग्राफ हटाया नहीं जा सका।")
        if isinstance(data, dict):
            return _format_dashboard_scope_message(data, lang=_preferred_language(context))
        if isinstance(data, str) and data.strip():
            if _preferred_language(context) == "hi" and not _contains_devanagari(data):
                return _voice("नवीनतम डैशबोर्ड ग्राफ हटा दिया है।")
            return _voice(data)
        return _say(context, en="Removed the latest dashboard graph.", hi="नवीनतम डैशबोर्ड ग्राफ हटा दिया है।")

    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

    params: Dict[str, Any] = {"active_only": "false"}
    dashboard_id = config.get("dashboard_id")
    if dashboard_id:
        params["dashboard_id"] = dashboard_id

    inventory, error = await _call_backend(
        context,
        "GET",
        "/dashboard/graphs",
        params=params,
        config=config,
    )
    if error:
        return _say(context, en=error, hi="डैशबोर्ड ग्राफ लोड नहीं हो पाए।")
    graphs = inventory.get("graphs") if isinstance(inventory, dict) else None
    if not graphs:
        return _say(context, en="There are no dashboard graphs to remove.", hi="हटाने के लिए कोई डैशबोर्ड ग्राफ नहीं है।")

    exact_match = None
    lower_query = query.lower()
    for graph in graphs:
        identifier = (graph.get("graph_id") or graph.get("id") or "").strip()
        title = (graph.get("title") or "").strip()
        if identifier == query or identifier.lower() == lower_query:
            exact_match = graph
            break
        if title and title.lower() == lower_query:
            exact_match = graph
            break

    if not exact_match:
        titles = [graph.get("title", "") for graph in graphs if graph.get("title")]
        close_titles = difflib.get_close_matches(query, titles, n=1, cutoff=0.6)
        if close_titles:
            chosen = close_titles[0]
            for graph in graphs:
                if (graph.get("title") or "").strip() == chosen:
                    exact_match = graph
                    break
        if not exact_match:
            payload = {"question": f"remove graph {query} from the dashboard"}
            if dashboard_id:
                payload["dashboard_id"] = dashboard_id
            fallback_data, fallback_error = await _call_backend(
                context,
                "POST",
                "/dashboard/graphs/query",
                payload=payload,
                config=config,
            )
            if fallback_error:
                return fallback_error
            if isinstance(fallback_data, dict):
                return _format_dashboard_scope_message(fallback_data, lang=_preferred_language(context))
            if isinstance(fallback_data, str) and fallback_data.strip():
                if _preferred_language(context) == "hi" and not _contains_devanagari(fallback_data):
                    return _voice(f"{query} ग्राफ हटाने की कोशिश की है।")
                return _voice(fallback_data)
            return _say(context, en=f"Attempted to remove {query}.", hi=f"{query} ग्राफ हटाने की कोशिश की है।")

    if not exact_match:
        return _say(context, en=f"No dashboard graph matched {query}.", hi=f"{query} नाम का कोई डैशबोर्ड ग्राफ नहीं मिला।")

    target_id = exact_match.get("graph_id") or exact_match.get("id")
    if not target_id:
        return _say(context, en="Unable to identify the selected dashboard graph.", hi="चुने हुए डैशबोर्ड ग्राफ की पहचान नहीं हो पाई।")

    _, error = await _call_backend(context, "DELETE", f"/dashboard/graphs/{target_id}", config=config)
    if error:
        return _say(context, en=error, hi="डैशबोर्ड ग्राफ हटाया नहीं जा सका।")
    title = exact_match.get("title") or target_id
    return _say(context, en=f"Removed dashboard graph {title}.", hi=f"डैशबोर्ड ग्राफ {title} हटा दिया है।")


@function_tool()
async def dashboard_update_scope(
    context: RunContext,  # type: ignore
    action: str,
    graph_identifiers: List[str],
) -> str:
    """
    Adjust dashboard graph scope (activate, deactivate, or exclusive).

    Use this when the user asks to enable, disable, or focus on specific dashboard graphs.
    """
    if not action or action.lower() not in {"activate", "deactivate", "exclusive"}:
        return _say(context, en="Scope action must be activate, deactivate, or exclusive.", hi="स्कोप एक्शन activate, deactivate, या exclusive होना चाहिए।")
    if not graph_identifiers:
        return _say(context, en="Specify at least one dashboard graph identifier.", hi="कम से कम एक डैशबोर्ड ग्राफ आईडी या नाम बताइए।")

    config, error = _resolve_backend_context(context)
    if error or not config:
        return _say(context, en=(error or "Backend configuration is unavailable."), hi=(error or "बैकएंड कॉन्फ़िगरेशन उपलब्ध नहीं है।"))

    payload: Dict[str, Any] = {
        "action": action.lower(),
        "graphs": graph_identifiers,
    }
    dashboard_id = config.get("dashboard_id")
    if dashboard_id:
        payload["dashboard_id"] = dashboard_id

    data, error = await _call_backend(
        context,
        "POST",
        "/dashboard/graphs/scope",
        payload=payload,
        config=config,
    )
    if error:
        return _say(context, en=error, hi="डैशबोर्ड स्कोप अपडेट नहीं हो पाया।")
    if not isinstance(data, dict):
        return _say(context, en="Dashboard scope updated.", hi="डैशबोर्ड स्कोप अपडेट कर दिया है।")
    updated = data.get("updated", 0)
    message = data.get("message") or "Dashboard scope updated."
    if _preferred_language(context) == "hi" and isinstance(message, str) and not _contains_devanagari(message):
        message = "डैशबोर्ड स्कोप अपडेट कर दिया है।"
    if updated:
        if _preferred_language(context) == "hi":
            message = f"{message} {updated} ग्राफ प्रभावित हुए।"
        else:
            message = f"{message} {updated} graph{'s' if updated != 1 else ''} affected."
    return _voice(message)
