from __future__ import annotations

import html
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import quote, quote_plus

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CAREERNET_API_URL = 'https://www.career.go.kr/cnet/openapi/getOpenApi'
CORPUS_SCHEMA_VERSION = '3'

RAW_COLUMNS = [
    'majorSeq', '계열', '학과명', '세부학과명', '학과개요', '흥미와적성', '학과특성',
    '관련고교교과', '진로탐색활동', '관련직업', '관련자격', '졸업후진출분야',
    '대학주요교과목',
]
CORPUS_TEXT_COLUMNS = RAW_COLUMNS[4:]
CHANNEL_COLUMNS = [
    '말뭉치_학업', '말뭉치_교과', '말뭉치_활동', '말뭉치_적성', '말뭉치_진로',
    '말뭉치_통합',
]


def validate_corpus_schema(
    df: pd.DataFrame,
    *,
    require_integrated: bool = True,
    require_raw: bool = True,
) -> None:
    """연구용 입력 DB가 말뭉치 원자료와 채널을 명시적으로 갖췄는지 확인합니다."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError('학과 말뭉치가 표 형식이 아닙니다.')
    required = ['majorSeq', '계열', '학과명']
    if require_raw:
        required.extend(column for column in RAW_COLUMNS if column not in required)
    if require_integrated:
        required.append('말뭉치_통합')
    missing = [column for column in dict.fromkeys(required) if column not in df.columns]
    if missing:
        raise ValueError(f'학과 말뭉치에 필요한 열이 없습니다: {", ".join(missing)}')


def clean_api_text(value: Any) -> str:
    if value is None:
        return ''
    text = html.unescape(str(value))
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _as_items(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _content_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and 'dataSearch' in payload:
        payload = payload.get('dataSearch')
    if isinstance(payload, dict) and 'content' in payload:
        payload = payload.get('content')
    return [item for item in _as_items(payload) if isinstance(item, dict)]


def _collect_values(value: Any, keys: Iterable[str]) -> List[str]:
    wanted = set(keys)
    found: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in wanted and not isinstance(child, (dict, list)):
                    text = clean_api_text(child)
                    if text:
                        found.append(text)
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return list(dict.fromkeys(found))


def _group_text(detail: Dict[str, Any], group: str, keys: Iterable[str]) -> str:
    return ' '.join(_collect_values(detail.get(group), keys))


def _join_unique(*values: Any) -> str:
    parts: List[str] = []
    seen = set()
    for value in values:
        if isinstance(value, (list, tuple)):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            text = clean_api_text(candidate)
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
    return ' '.join(parts)


def build_corpus_channels(row: Dict[str, Any]) -> Dict[str, str]:
    academic = _join_unique(row.get('학과개요'), row.get('학과특성'), row.get('대학주요교과목'))
    subjects = _join_unique(row.get('관련고교교과'), row.get('대학주요교과목'))
    activities = _join_unique(row.get('진로탐색활동'))
    aptitude = _join_unique(row.get('흥미와적성'))
    career = _join_unique(row.get('졸업후진출분야'), row.get('관련직업'), row.get('관련자격'))
    # 통합 문서는 채널을 재결합하지 않고 원자료 필드를 한 번씩 읽습니다.
    # 따라서 대학주요교과목이 학업·교과 채널에 포함되더라도 통합에서는 한 번만 남습니다.
    combined = _join_unique(*(row.get(column) for column in ['학과명', '세부학과명', *CORPUS_TEXT_COLUMNS]))
    # ``말뭉치``만 있던 구버전 파일은 앱에서 계속 읽을 수 있게 보존합니다.
    # 원자료 값이 하나라도 있으면 임의의 레거시 텍스트를 섞지 않습니다.
    if not any(clean_api_text(row.get(column)) for column in CORPUS_TEXT_COLUMNS):
        combined = _join_unique(row.get('말뭉치'), row.get('학과명'), row.get('세부학과명'))
    return {
        '말뭉치_학업': academic,
        '말뭉치_교과': subjects,
        '말뭉치_활동': activities,
        '말뭉치_적성': aptitude,
        '말뭉치_진로': career,
        '말뭉치_통합': combined,
        # 기존 파일과 분석 코드의 하위 호환을 위해 유지합니다.
        '말뭉치': combined,
    }


def redact_api_key(message: Any, api_key: str = '') -> str:
    """예외·로그에 남는 커리어넷 URL의 apiKey 값을 제거합니다."""
    text = clean_api_text(message)
    text = re.sub(
        r'([?&](?:apiKey|api_key)=)[^&#\s]*',
        r'\1[REDACTED]',
        text,
        flags=re.IGNORECASE,
    )
    secret = str(api_key or '').strip()
    if secret:
        candidates = {secret, quote(secret, safe=''), quote_plus(secret)}
        for value in sorted(candidates, key=len, reverse=True):
            if value:
                text = text.replace(value, '[REDACTED]')
    return text


def format_careernet_error(exc: BaseException, api_key: str = '') -> str:
    """사용자에게 보여 줄 수 있는 안전한 커리어넷 요청 오류를 만듭니다."""
    if isinstance(exc, requests.exceptions.SSLError):
        return (
            '커리어넷 HTTPS 인증서 검증에 실패했습니다. API 키 오류가 아니라 '
            'Windows 시스템 인증서 저장소 또는 학교 네트워크 인증서를 확인해 주세요.'
        )
    return redact_api_key(f'{type(exc).__name__}: {exc}', api_key)


_TRUSTSTORE_CONFIGURED = False


def configure_system_certificate_store() -> str:
    """가능한 환경에서 OS 인증서 저장소를 requests의 SSL에 연결합니다."""
    global _TRUSTSTORE_CONFIGURED
    if _TRUSTSTORE_CONFIGURED:
        return 'truststore'
    try:
        import truststore
    except ImportError:
        # 일반 환경에서는 requests의 기본 CA 검증을 그대로 사용합니다.
        return 'requests-default'
    truststore.inject_into_ssl()
    _TRUSTSTORE_CONFIGURED = True
    return 'truststore'


def enrich_existing_corpus(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    enriched = df.copy().fillna('')
    for column in RAW_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = ''
    channels = [build_corpus_channels(row.to_dict()) for _, row in enriched.iterrows()]
    channel_df = pd.DataFrame(channels, index=enriched.index)
    for column in channel_df.columns:
        enriched[column] = channel_df[column]
    if '수집시각' not in enriched.columns:
        enriched['수집시각'] = ''
    enriched['스키마버전'] = CORPUS_SCHEMA_VERSION
    if '수집오류' not in enriched.columns:
        enriched['수집오류'] = ''
    return enriched


def _make_session() -> requests.Session:
    configure_system_certificate_store()
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=('GET',),
    )
    session.mount('https://', HTTPAdapter(max_retries=retry))
    session.headers.update({'User-Agent': 'StudentRecordExplorer/3.0'})
    return session


def _request_json(session: requests.Session, api_key: str, **params: Any) -> Dict[str, Any]:
    query = {
        'apiKey': api_key,
        'svcType': 'api',
        'contentType': 'json',
        'gubun': 'univ_list',
        **params,
    }
    try:
        response = session.get(CAREERNET_API_URL, params=query, timeout=(5, 25))
        response.raise_for_status()
        payload = response.json()
        result = payload.get('result') if isinstance(payload, dict) else None
        if isinstance(result, dict) and str(result.get('code', '0')) not in {'0', ''}:
            raise RuntimeError(clean_api_text(result.get('message')) or '커리어넷 API 오류')
        return payload
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(format_careernet_error(exc, api_key)) from exc
    except Exception as exc:
        raise RuntimeError(format_careernet_error(exc, api_key)) from exc


def fetch_major_list(
    api_key: str,
    session: Optional[requests.Session] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[Dict[str, str]]:
    session = session or _make_session()
    page = 1
    per_page = 100
    majors: Dict[str, Dict[str, str]] = {}
    while True:
        payload = _request_json(
            session,
            api_key,
            svcCode='MAJOR',
            thisPage=page,
            perPage=per_page,
        )
        items = _content_items(payload)
        if not items:
            break
        for item in items:
            major_seq = clean_api_text(item.get('majorSeq'))
            if not major_seq:
                continue
            majors[major_seq] = {
                'majorSeq': major_seq,
                '계열': clean_api_text(item.get('lClass')),
                '학과명': clean_api_text(item.get('mClass')),
                '세부학과명': clean_api_text(item.get('facilName')),
            }
        total_values = _collect_values(payload.get('dataSearch', payload), ('totalCount',))
        total = int(total_values[0]) if total_values and total_values[0].isdigit() else 0
        if progress_callback:
            denominator = max(total, len(majors), 1)
            progress_callback(min(len(majors) / denominator, 1.0), f'학과 목록 수집 중 · {len(majors)}/{total or "?"}개')
        if len(items) < per_page or (total and len(majors) >= total):
            break
        page += 1
        if page > 100:
            raise RuntimeError('학과 목록 페이지 수가 예상 범위를 초과했습니다.')
    return list(majors.values())


def parse_major_detail(base: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    items = _content_items(payload)
    detail = items[0] if items else {}
    row: Dict[str, Any] = {
        **base,
        '학과명': clean_api_text(detail.get('major')) or base.get('학과명', ''),
        '세부학과명': clean_api_text(detail.get('department')) or base.get('세부학과명', ''),
        '학과개요': clean_api_text(detail.get('summary')),
        '흥미와적성': clean_api_text(detail.get('interest')),
        '학과특성': clean_api_text(detail.get('property')),
        '관련고교교과': _group_text(detail, 'relate_subject', ('subject_name', 'subject_description')),
        '진로탐색활동': _group_text(detail, 'career_act', ('act_name', 'act_description')),
        '관련직업': _join_unique(detail.get('job')),
        '관련자격': _join_unique(detail.get('qualifications')),
        '졸업후진출분야': _group_text(detail, 'enter_field', ('gradeuate', 'graduate', 'description')),
        '대학주요교과목': _group_text(detail, 'main_subject', ('SBJECT_NM', 'SBJECT_SUMRY')),
        '수집오류': '',
    }
    row.update(build_corpus_channels(row))
    row['수집시각'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row['스키마버전'] = CORPUS_SCHEMA_VERSION
    return row


def collect_careernet_corpus(
    api_key: str,
    existing: Optional[pd.DataFrame] = None,
    refresh_all: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    if not api_key.strip():
        raise ValueError('커리어넷 API 키를 입력해 주세요.')
    session = _make_session()
    if progress_callback:
        progress_callback(0.01, '커리어넷 학과 목록을 요청하는 중입니다.')
    majors = fetch_major_list(
        api_key.strip(),
        session=session,
        progress_callback=(
            (lambda fraction, message: progress_callback(0.01 + fraction * 0.09, message))
            if progress_callback else None
        ),
    )
    if not majors:
        raise RuntimeError('커리어넷에서 학과 목록을 받지 못했습니다.')

    existing_rows: Dict[str, Dict[str, Any]] = {}
    if existing is not None and not existing.empty and 'majorSeq' in existing.columns:
        existing_rows = {
            str(row.get('majorSeq', '')): row.to_dict()
            for _, row in enrich_existing_corpus(existing).iterrows()
        }

    rows: List[Dict[str, Any]] = []
    total = len(majors)
    for index, base in enumerate(majors, start=1):
        major_seq = base['majorSeq']
        old = existing_rows.get(major_seq)
        essential_fields = ('학과개요', '흥미와적성', '학과특성', '관련고교교과', '진로탐색활동', '대학주요교과목')
        old_is_complete = bool(old) and not clean_api_text(old.get('수집오류')) and all(
            clean_api_text(old.get(field)) for field in essential_fields
        )
        if old_is_complete and not refresh_all:
            row = old
        else:
            try:
                payload = _request_json(
                    session,
                    api_key.strip(),
                    svcCode='MAJOR_VIEW',
                    majorSeq=major_seq,
                )
                row = parse_major_detail(base, payload)
            except Exception as exc:
                if old:
                    row = {**old, '수집오류': format_careernet_error(exc, api_key)}
                else:
                    row = {**base, '수집오류': format_careernet_error(exc, api_key)}
                    row.update(build_corpus_channels(row))
                    row['수집시각'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    row['스키마버전'] = CORPUS_SCHEMA_VERSION
        rows.append(row)
        if progress_callback:
            progress_callback(
                0.10 + 0.90 * index / total,
                f'학과 상세정보 수집 중 · {index}/{total}개 · {base.get("학과명", "")}',
            )
        time.sleep(0.01)
    return enrich_existing_corpus(pd.DataFrame(rows))


def corpus_quality(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {'total': 0, 'errors': 0, 'complete': 0, 'updated_at': ''}
    data = df.fillna('')
    error_series = data.get('수집오류', pd.Series('', index=data.index)).astype(str).str.strip()
    updated = data.get('수집시각', pd.Series('', index=data.index)).astype(str)
    return {
        'total': len(data),
        'errors': int((error_series != '').sum()),
        'complete': int((error_series == '').sum()),
        'updated_at': max((value for value in updated if value), default=''),
        'field_counts': {
            column: int((data.get(column, pd.Series('', index=data.index)).astype(str).str.strip() != '').sum())
            for column in RAW_COLUMNS[4:]
        },
    }
