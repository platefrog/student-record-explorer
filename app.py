# -*- coding: utf-8 -*-
from __future__ import annotations
import io, re, html, os, sqlite3, platform, time, json, zipfile, shutil, threading
from pathlib import Path
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyBboxPatch
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from careernet_corpus import (
    collect_careernet_corpus,
    corpus_quality,
    enrich_existing_corpus,
)

try:
    from kiwipiepy import Kiwi
    KIWI_OK = True
    KIWI_IMPORT_ERROR = ''
except Exception as exc:
    Kiwi = None
    KIWI_OK = False
    KIWI_IMPORT_ERROR = f'{type(exc).__name__}: {exc}'
KIWI_RUNTIME_ERROR = ''

MECAB_BACKEND = ''
try:
    import mecab_ko as MecabKo
    MECAB_OK = True
    MECAB_BACKEND = 'mecab_ko'
except Exception:
    MecabKo = None
    try:
        from mecab import MeCab as PythonMecab
        MECAB_OK = True
        MECAB_BACKEND = 'python_mecab_ko'
    except Exception:
        PythonMecab = None
        MECAB_OK = False

APP_DIR = Path(__file__).resolve().parent
try:
    APP_VERSION = (APP_DIR / 'VERSION').read_text(encoding='utf-8').strip() or 'development'
except Exception:
    APP_VERSION = 'development'
BUNDLED_DATA_DIR = APP_DIR / 'data'
DATA_DIR = Path(os.environ.get('SRE_DATA_DIR', str(BUNDLED_DATA_DIR))).expanduser()
DB_PATH = DATA_DIR / 'major_corpus.db'
STOPWORDS_PATH = DATA_DIR / 'stopwords.txt'
SYNONYMS_PATH = DATA_DIR / 'synonyms.txt'
STUDENT_CACHE_PATH = DATA_DIR / 'student_cache_latest.db'

DEFAULT_STOPWORDS = set('''학생 활동 수업 참여 통해 대한 관련 내용 주제 과정 보고서 탐구 발표 작성 자신 능력 모습 학습 이해 설명 자료 분석 의견 생각 학교 교사 진로 학과 계열 교육 흥미 적성 직업 특성 개요 주요 분야 졸업 진출 사항 기록'''.split())
NAME_RE = re.compile(r'^[가-힣]{2,5}$')
NO_RE = re.compile(r'^\d{1,3}$')
STUDENT_ID_COLUMNS = ['학년', '반', '번호', '성명']


def ensure_files():
    """쓰기 가능한 사용자 데이터 폴더를 만들고 기본 자료를 최초 1회 복사합니다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    defaults = {
        STOPWORDS_PATH: '\n'.join(sorted(DEFAULT_STOPWORDS)),
        SYNONYMS_PATH: 'AI=인공지능\nSW=소프트웨어\n코딩=프로그래밍\nDB=데이터베이스\n',
    }
    for destination, fallback in defaults.items():
        if destination.exists():
            continue
        source = BUNDLED_DATA_DIR / destination.name
        if source.exists() and source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        else:
            destination.write_text(fallback, encoding='utf-8')
    bundled_db = BUNDLED_DATA_DIR / DB_PATH.name
    if not DB_PATH.exists() and bundled_db.exists() and bundled_db.resolve() != DB_PATH.resolve():
        shutil.copy2(bundled_db, DB_PATH)


def render_desktop_controls() -> None:
    """설치형 배포판에서 백그라운드 로컬 서버를 종료할 수 있게 합니다."""
    if os.environ.get('SRE_DESKTOP_MODE') != '1':
        return
    st.sidebar.divider()
    st.sidebar.caption(f'로컬 저장 위치: {DATA_DIR}')
    if st.sidebar.button('프로그램 종료', key='desktop_exit_request'):
        st.session_state['desktop_exit_confirm'] = True
    if st.session_state.get('desktop_exit_confirm'):
        st.sidebar.warning('학생부 탐색기를 종료할까요? 저장 중인 작업이 없는지 확인해 주세요.')
        confirm_col, cancel_col = st.sidebar.columns(2)
        if confirm_col.button('종료', type='primary', key='desktop_exit_confirm_button'):
            st.sidebar.success('프로그램을 종료합니다. 이 탭을 닫아도 됩니다.')

            def stop_process() -> None:
                time.sleep(0.8)
                os._exit(0)

            threading.Thread(target=stop_process, daemon=True).start()
            st.stop()
        if cancel_col.button('취소', key='desktop_exit_cancel'):
            st.session_state['desktop_exit_confirm'] = False
            st.rerun()


def clean(x: Any) -> str:
    if x is None or pd.isna(x): return ''
    s = html.unescape(str(x)).replace('\u3000', ' ')
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_school_number(value: Any) -> str:
    """학년·반·번호 값을 비교하기 쉬운 문자열로 정규화합니다."""
    value = clean(value)
    match = re.fullmatch(r'(\d+)(?:\.0+)?', value)
    return str(int(match.group(1))) if match else value


def extract_grade_class(raw: pd.DataFrame, file_name: str = '') -> Tuple[str, str]:
    """파일명과 엑셀 상단에서 학년·반을 찾습니다.

    지원 예: ``3-1반 창체.xlsx``, ``3학년 1반 세특.xlsx``.
    """
    file_text = Path(file_name or '').stem
    top_values = raw.head(25).fillna('').astype(str).values.ravel().tolist()
    sheet_text = ' '.join(clean(value) for value in top_values if clean(value))

    combined_patterns = [r'(?<!\d)(\d{1,2})\s*학년(?!도)\s*[:：]?\s*(\d{1,2})\s*반']
    for text in [file_text, sheet_text]:
        for pattern in combined_patterns:
            match = re.search(pattern, text)
            if match:
                return normalize_school_number(match.group(1)), normalize_school_number(match.group(2))
    file_match = re.search(r'(?<!\d)(\d{1,2})\s*[-_]\s*(\d{1,2})\s*반?', file_text)
    if file_match:
        return normalize_school_number(file_match.group(1)), normalize_school_number(file_match.group(2))

    grade = ''
    class_no = ''
    for text in [file_text, sheet_text]:
        if not grade:
            match = re.search(r'(?:학년(?!도)\s*[:：]?\s*(\d{1,2})|(?<!\d)(\d{1,2})\s*학년(?!도))', text)
            if match:
                grade = normalize_school_number(match.group(1) or match.group(2))
        if not class_no:
            match = re.search(r'(?:반\s*[:：]?\s*(\d{1,2})|(?<!\d)(\d{1,2})\s*반)', text)
            if match:
                class_no = normalize_school_number(match.group(1) or match.group(2))
    return grade, class_no


def ensure_student_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """이전 버전 캐시에도 학년·반 식별 열을 보충합니다."""
    df = df.copy()
    for column in STUDENT_ID_COLUMNS:
        if column not in df.columns:
            df[column] = ''
        df[column] = df[column].fillna('').map(normalize_school_number if column != '성명' else clean)
    identity = STUDENT_ID_COLUMNS
    remaining = [column for column in df.columns if column not in identity]
    return df[identity + remaining]


def student_mask(df: pd.DataFrame, student: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column in STUDENT_ID_COLUMNS:
        if column in df.columns:
            expected = normalize_school_number(student.get(column, '')) if column != '성명' else clean(student.get(column, ''))
            actual = df[column].fillna('').map(normalize_school_number if column != '성명' else clean)
            mask &= actual == expected
    return mask


def student_label(student: pd.Series) -> str:
    grade = normalize_school_number(student.get('학년', ''))
    class_no = normalize_school_number(student.get('반', ''))
    number = normalize_school_number(student.get('번호', ''))
    name = clean(student.get('성명', ''))
    parts = []
    if grade:
        parts.append(f'{grade}학년')
    if class_no:
        parts.append(f'{class_no}반')
    if number:
        parts.append(f'{number}번')
    parts.append(name)
    return ' '.join(parts)


def read_stopwords(extra='') -> set[str]:
    ensure_files(); words = set(DEFAULT_STOPWORDS)
    for p in [STOPWORDS_PATH]:
        try:
            for line in p.read_text(encoding='utf-8').splitlines():
                line=line.strip()
                if line and not line.startswith('#'): words.add(line)
        except Exception: pass
    for w in re.split(r'[,\n]', extra or ''):
        if w.strip(): words.add(w.strip())
    return words


def read_synonyms() -> dict[str,str]:
    ensure_files(); mp={}
    try:
        for line in SYNONYMS_PATH.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line or line.startswith('#'): continue
            sep = '=' if '=' in line else (',' if ',' in line else None)
            if sep:
                a,b=line.split(sep,1); mp[a.strip()]=b.strip()
    except Exception: pass
    return mp


def dictionary_stopword_candidates(
    cache: Any,
    current_stopwords: set[str],
    min_student_ratio: float = 0.35,
) -> pd.DataFrame:
    """현재 학생부 캐시에서 여러 학생에게 반복되는 불용어 검토 후보를 찾습니다.

    결과를 사전에 자동 반영하지는 않습니다. 학생별 상위 빈도어만 저장된 캐시를
    이용하므로, 후보를 원문 맥락과 함께 검토하기 위한 보조 지표입니다.
    """
    columns = ['단어', '등장 학생', '학생 비율(%)', '전체 빈도']
    if not isinstance(cache, dict):
        return pd.DataFrame(columns=columns)
    freq = cache.get('freq', pd.DataFrame())
    records = cache.get('records', pd.DataFrame())
    if not isinstance(freq, pd.DataFrame) or freq.empty or not {'단어', '빈도'}.issubset(freq.columns):
        return pd.DataFrame(columns=columns)

    freq = ensure_student_identity_columns(freq).copy()
    freq['단어'] = freq['단어'].fillna('').astype(str).str.strip()
    freq['빈도'] = pd.to_numeric(freq['빈도'], errors='coerce').fillna(0)
    freq = freq[(freq['단어'].str.len() >= 2) & ~freq['단어'].isin(current_stopwords)]
    if freq.empty:
        return pd.DataFrame(columns=columns)

    if isinstance(records, pd.DataFrame) and not records.empty:
        student_count = len(ensure_student_identity_columns(records)[STUDENT_ID_COLUMNS].drop_duplicates())
    else:
        student_count = len(freq[STUDENT_ID_COLUMNS].drop_duplicates())
    if student_count == 0:
        return pd.DataFrame(columns=columns)

    unique_students = freq.drop_duplicates(STUDENT_ID_COLUMNS + ['단어'])
    document_frequency = unique_students.groupby('단어').size().rename('등장 학생')
    total_frequency = freq.groupby('단어')['빈도'].sum().rename('전체 빈도')
    result = pd.concat([document_frequency, total_frequency], axis=1).reset_index()
    result['학생 비율(%)'] = (result['등장 학생'] / student_count * 100).round(1)
    minimum_students = max(2, int(student_count * min_student_ratio + 0.9999))
    result = result[result['등장 학생'] >= minimum_students]
    result['전체 빈도'] = result['전체 빈도'].astype(int)
    return result[columns].sort_values(
        ['학생 비율(%)', '전체 빈도', '단어'], ascending=[False, False, True]
    ).reset_index(drop=True)


def apply_syn(t: str, syn: dict[str,str]) -> str:
    return syn.get(t, syn.get(t.upper(), syn.get(t.lower(), t)))


def normalize_token_surface(t: str) -> str:
    suffixes=['으로부터','로부터','에서는','에게는','이라는','라는','으로','에서','에게','부터','까지','하고','하며','하여','해서','하는','하게','되어','되는','되고','들을','에도','에는','로서','보다','처럼','은','는','이','가','을','를','의','에','와','과','도','만','로','고','며']
    changed=True
    while changed and len(t)>=3:
        changed=False
        for suf in suffixes:
            if t.endswith(suf) and len(t)-len(suf)>=2:
                t=t[:-len(suf)]; changed=True; break
    return t

@st.cache_resource(show_spinner=False)
def get_kiwi():
    global KIWI_RUNTIME_ERROR
    if not KIWI_OK:
        return None
    try:
        kiwi = Kiwi()
        KIWI_RUNTIME_ERROR = ''
        return kiwi
    except Exception as exc:
        KIWI_RUNTIME_ERROR = f'{type(exc).__name__}: {exc}'
        return None


@st.cache_resource(show_spinner=False)
def get_mecab():
    if not MECAB_OK:
        return None
    if MECAB_BACKEND == 'mecab_ko':
        return MecabKo.Tagger()
    return PythonMecab()


def analyzer_name(setting: Any) -> str:
    if isinstance(setting, bool):
        return 'Kiwi' if setting else '간이 토큰화'
    text = str(setting or '').strip().lower()
    if 'mecab' in text or '메캡' in text:
        return 'MeCab'
    if 'kiwi' in text or '키위' in text:
        return 'Kiwi'
    return '간이 토큰화'


def analyzer_available(setting: Any) -> bool:
    name = analyzer_name(setting)
    if name == 'Kiwi':
        return get_kiwi() is not None
    if name == 'MeCab':
        return MECAB_OK
    return True


def analyzer_unavailable_reason(setting: Any) -> str:
    name = analyzer_name(setting)
    if name == 'Kiwi':
        if KIWI_IMPORT_ERROR:
            return KIWI_IMPORT_ERROR
        if KIWI_RUNTIME_ERROR:
            return KIWI_RUNTIME_ERROR
    elif name == 'MeCab' and not MECAB_OK:
        return 'MeCab is not installed.'
    return ''


def tokenize(text: str, stop: set[str], syn: dict[str,str], min_len=2, analyzer='Kiwi') -> List[str]:
    text=clean(text); out=[]
    mode = analyzer_name(analyzer)
    if mode == 'Kiwi':
        kiwi = get_kiwi()
        if kiwi is None:
            raise RuntimeError('Kiwi 형태소 분석기를 사용할 수 없습니다.')
        try:
            for tok in kiwi.tokenize(text):
                if tok.tag.startswith('N') or tok.tag in {'SL','SH'}:
                    f=apply_syn(tok.form.strip(), syn)
                    if len(f)>=min_len and f not in stop: out.append(f)
            return out
        except Exception:
            pass
    elif mode == 'MeCab':
        mecab = get_mecab()
        if mecab is None:
            raise RuntimeError('MeCab 형태소 분석기가 설치되어 있지 않습니다.')
        try:
            if MECAB_BACKEND == 'mecab_ko':
                parsed = mecab.parse(text) or ''
                pairs = []
                for line in parsed.splitlines():
                    if line == 'EOS' or '\t' not in line:
                        continue
                    surface, features = line.split('\t', 1)
                    pairs.append((surface, features.split(',', 1)[0]))
            else:
                pairs = mecab.pos(text)
            for surface, tag in pairs:
                if str(tag).startswith('N') or tag in {'SL', 'SH'}:
                    f = apply_syn(str(surface).strip(), syn)
                    if len(f) >= min_len and f not in stop:
                        out.append(f)
            return out
        except Exception as exc:
            raise RuntimeError(f'MeCab 형태소 분석 중 오류가 발생했습니다: {exc}') from exc
    for t in re.findall(r'[가-힣A-Za-z]{%d,}' % min_len, text):
        t=apply_syn(normalize_token_surface(t), syn)
        if len(t)>=min_len and t not in stop: out.append(t)
    return out


def tokenized(text, stop, syn, min_len, analyzer): return ' '.join(tokenize(text, stop, syn, min_len, analyzer))


@st.cache_data(show_spinner=False, persist='disk', max_entries=12)
def prepare_major_index(
    corpus_documents: Tuple[str, ...],
    stop_words: Tuple[str, ...],
    synonym_items: Tuple[Tuple[str, str], ...],
    min_len: int,
    analyzer: Any,
):
    """학과 말뭉치를 한 번만 형태소 분석하고 TF-IDF 인덱스로 재사용합니다."""
    stop = set(stop_words)
    syn = dict(synonym_items)
    docs = [tokenized(text, stop, syn, min_len, analyzer) for text in corpus_documents]
    if not any(docs):
        return None, None, None
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix, vectorizer.get_feature_names_out()


def split_record_sentences(text: str) -> List[str]:
    """생기부 원문을 근거 확인용 짧은 단위로 나눕니다.

    생기부 문장은 마침표 없이 긴 경우가 있어 문장부호와 줄바꿈을 함께 사용합니다.
    너무 긴 조각은 화면에서 읽기 좋게 다시 잘라 줍니다.
    """
    text = clean(text)
    if not text:
        return []
    parts = re.split(r'(?<=[.!?。])\s+|[\n\r]+', text)
    out: List[str] = []
    for part in parts:
        part = clean(part)
        if not part:
            continue
        if len(part) <= 260:
            out.append(part)
        else:
            chunks = re.split(r'(?<=다[. ]|함[. ]|음[. ]|임[. ])', part)
            buf = ''
            for ch in chunks:
                ch = clean(ch)
                if not ch:
                    continue
                if len(buf) + len(ch) <= 260:
                    buf = (buf + ' ' + ch).strip()
                else:
                    if buf:
                        out.append(buf)
                    buf = ch
            if buf:
                out.append(buf)
    return out


def keyword_evidence(row: pd.Series, keyword: str, stop: set[str], syn: dict[str, str], min_len: int, use_kiwi: bool) -> pd.DataFrame:
    """선택 키워드가 원문 어디에서 나왔는지 역추적합니다.

    원문에 동일 표면형이 직접 등장하지 않아도, 토큰화/표현통일 결과가 keyword와 같으면 근거로 잡습니다.
    예: 원문 'AI' → 표현통일 '인공지능'일 때, keyword가 '인공지능'이면 해당 문장을 찾습니다.
    """
    rows = []
    for source in ['창체', '교과세특', '행발']:
        source_text = str(row.get(source, '') or '')
        for idx, sent in enumerate(split_record_sentences(source_text), start=1):
            toks = tokenize(sent, stop, syn, min_len, use_kiwi)
            count = toks.count(keyword)
            direct_count = len(re.findall(re.escape(keyword), sent, flags=re.IGNORECASE)) if keyword else 0
            hit_count = max(count, direct_count)
            if hit_count > 0:
                highlighted = sent
                # 직접 표면형이 있는 경우만 우선 강조합니다. 표현통일로 잡힌 경우는 문장 전체를 근거로 제시합니다.
                try:
                    highlighted = re.sub(f'({re.escape(keyword)})', r'**\1**', highlighted, flags=re.IGNORECASE)
                except Exception:
                    pass
                rows.append({
                    '출처': source,
                    '문장번호': idx,
                    '등장횟수': hit_count,
                    '원문': sent,
                    '강조원문': highlighted,
                })
    return pd.DataFrame(rows)


def evidence_summary(evd: pd.DataFrame) -> pd.DataFrame:
    if evd.empty:
        return pd.DataFrame(columns=['출처', '등장횟수', '근거문장수'])
    return evd.groupby('출처', as_index=False).agg(등장횟수=('등장횟수','sum'), 근거문장수=('원문','count'))



def render_help(title: str, body: str):
    """학술용어 설명 버튼입니다. Streamlit 버전에 따라 popover 또는 expander로 표시합니다."""
    if hasattr(st, 'popover'):
        with st.popover(f'❔ {title}'):
            st.markdown(body)
    else:
        with st.expander(f'❔ {title}', expanded=False):
            st.markdown(body)


def apply_ui_style():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 3rem; max-width: 1320px;}
        h1 {letter-spacing: -0.04em;}
        h2, h3 {letter-spacing: -0.035em;}
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            font-size: 1.35rem !important;
        }
        div[data-testid="stMetricValue"] {font-size: 1.7rem;}
        div[data-testid="stPopover"] {
            width: min(100%, 17rem);
            margin: .2rem 0 .65rem 0;
        }
        div[data-testid="stPopover"] > button {
            width: 100%;
            min-height: 2.65rem;
            justify-content: center;
            border-radius: 10px;
            font-size: .92rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .sre-card {
            border: 1px solid rgba(148,163,184,.22);
            border-radius: 16px;
            background: rgba(148,163,184,.10);
            padding: 1.1rem 1.15rem;
            margin: .55rem 0 1rem 0;
            box-shadow: 0 8px 28px rgba(0,0,0,.10);
        }
        .sre-status-item {
            border: 1px solid rgba(148,163,184,.18);
            border-radius: 14px;
            background: rgba(148,163,184,.10);
            padding: .85rem .95rem;
        }
        .sre-status-label {
            color: inherit;
            opacity: .78;
            font-size: .84rem;
            font-weight: 700;
            margin-bottom: .25rem;
        }
        .sre-status-value {
            color: inherit;
            font-size: 1.52rem;
            font-weight: 900;
            letter-spacing: -0.04em;
        }
        .sre-section-title {
            font-size: 1.28rem;
            font-weight: 800;
            margin: .2rem 0 .25rem 0;
            letter-spacing: -0.03em;
        }
        .sre-section-caption {
            color: inherit;
            opacity: .78;
            font-size: .92rem;
            margin-bottom: .75rem;
        }
        .sre-gauge-wrap {border:1px solid rgba(148,163,184,.22); border-radius:14px; padding:1rem; margin-bottom:.8rem; background:rgba(148,163,184,.10);}
        .sre-gauge-label {font-size:.88rem; color:inherit; opacity:.78; margin-bottom:.35rem;}
        .sre-gauge-value {font-size:2.2rem; font-weight:900; color:inherit; letter-spacing:-.04em;}
        .sre-bar-bg {height: 13px; background:rgba(148,163,184,.22); border-radius:999px; overflow:hidden; margin:.65rem 0 .35rem;}
        .sre-bar-fill {height:13px; border-radius:999px; background:linear-gradient(90deg,#60A5FA,#34D399);}
        .sre-small {font-size:.82rem; color:inherit; opacity:.78;}
        .sre-pill {display:inline-block; padding:.22rem .55rem; border-radius:999px; background:rgba(59,130,246,.13); border:1px solid rgba(59,130,246,.25); color:inherit; font-size:.78rem; margin:.12rem;}
        .sre-evidence-scroll {
            max-height: 540px; overflow-y: auto; padding: .25rem .65rem .25rem .15rem;
            border-left: 1px solid rgba(148,163,184,.22);
        }
        .sre-evidence-card {
            border-left: 3px solid #60A5FA;
            background: rgba(148,163,184,.10);
            border-radius: 10px;
            padding: .8rem .9rem;
            margin-bottom: .8rem;
        }
        .sre-evidence-meta {font-weight:800; color:inherit; margin-bottom:.35rem;}
        .sre-evidence-text {color:inherit; line-height:1.75; font-size:.95rem; opacity:.92;}
        .sre-keyword-highlight {
            background: linear-gradient(120deg, #FDE68A, #FBBF24);
            color: #422006;
            border-radius: 4px;
            padding: .05rem .22rem;
            font-weight: 900;
            box-shadow: 0 0 0 1px rgba(251,191,36,.25);
        }
        .sre-selected-keyword {
            display: inline-block;
            margin: 0 0 .7rem 0;
            padding: .28rem .65rem;
            border-radius: 999px;
            background: rgba(251,191,36,.14);
            border: 1px solid rgba(251,191,36,.35);
            color: inherit;
            font-size: .82rem;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_theme_mode(mode: str) -> None:
    is_dark = (mode or 'dark') == 'dark'
    if is_dark:
        st.markdown(
            """
            <style>
            .stApp, [data-testid="stAppViewContainer"] {background-color:#0B1220; color:#E5E7EB;}
            [data-testid="stHeader"] {background:rgba(11,18,32,.88);}
            [data-testid="stSidebar"] {background-color:#111827; color:#E5E7EB;}
            [data-testid="stSidebar"] * {color:#E5E7EB;}
            div[data-baseweb="input"] > div,
            div[data-baseweb="select"] > div,
            div[data-baseweb="textarea"] > div {
                background-color:#0F172A;
                color:#E5E7EB;
                border-color:rgba(148,163,184,.32);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp, [data-testid="stAppViewContainer"] {background-color:#F8FAFC; color:#0F172A;}
            [data-testid="stHeader"] {background:rgba(248,250,252,.9);}
            [data-testid="stSidebar"] {background-color:#EEF2F7; color:#0F172A;}
            [data-testid="stSidebar"] * {color:#0F172A;}
            </style>
            """,
            unsafe_allow_html=True,
        )


def card_open(title: str, caption: str = ''):
    # Streamlit 위젯을 HTML div로 감싸면 빈 테두리 박스가 생길 수 있어,
    # 섹션 제목/설명만 렌더링합니다.
    st.markdown(f'<div class="sre-section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="sre-section-caption">{caption}</div>', unsafe_allow_html=True)


def card_close():
    return None


def render_status_card(student_count: int, scope: str, changche_count: int, setuk_count: int, haengbal_count: int):
    html_block = f"""
    <div class="sre-card">
      <div class="sre-section-title">분석 현황</div>
      <div class="sre-section-caption">현재 업로드된 학생부 자료와 분석 범위입니다.</div>
      <div style="display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .85rem;">
        <div class="sre-status-item">
          <div class="sre-status-label">학생 수</div>
          <div class="sre-status-value">{student_count:,}명</div>
        </div>
        <div class="sre-status-item">
          <div class="sre-status-label">분석 범위</div>
          <div class="sre-status-value">{scope}</div>
        </div>
        <div class="sre-status-item">
          <div class="sre-status-label">창체 보유</div>
          <div class="sre-status-value">{changche_count:,}명</div>
        </div>
        <div class="sre-status-item">
          <div class="sre-status-label">교과세특 / 행발</div>
          <div class="sre-status-value">{setuk_count:,} / {haengbal_count:,}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def similarity_level(score: float, thresholds: Dict[str, float]) -> str:
    """전체 학생×학과 비교 분포를 기준으로 연계 수준을 판정합니다."""
    if score >= thresholds.get('fit', 1.0):
        return '적합'
    if score >= thresholds.get('high', 1.0):
        return '높은 연계'
    if score >= thresholds.get('medium', 1.0):
        return '보통 연계'
    return '낮은 연계'


def render_similarity_dashboard(
    metrics: Dict[str, Any],
    thresholds: Dict[str, float],
    gap_count: int = 0,
):
    cosine_score = float(metrics.get('cosine', 0.0))
    cosine_pct = cosine_score * 100
    jac_pct = float(metrics.get('jaccard', 0.0)) * 100
    common_count = int(metrics.get('common_count', 0))
    student_count = int(metrics.get('student_count', 0))
    target_count = int(metrics.get('target_count', 0))
    level = similarity_level(cosine_score, thresholds)
    st.markdown(
        f'''
        <div class="sre-gauge-wrap">
          <div class="sre-gauge-label">학생부 ↔ 목표 학과 TF-IDF 코사인 유사도</div>
          <div class="sre-gauge-value">{cosine_pct:.1f}%</div>
          <div class="sre-bar-bg"><div class="sre-bar-fill" style="width:{min(max(cosine_pct,0),100):.1f}%"></div></div>
          <div class="sre-small">연계 수준: <b>{level}</b> · 키워드 교집합 비율 {jac_pct:.1f}%</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('공통 키워드', f'{common_count:,}개')
    m2.metric('학생부 고유', f'{max(student_count-common_count,0):,}개')
    m3.metric('학과 고유', f'{max(target_count-common_count,0):,}개')
    m4.metric('보완 키워드', f'{gap_count:,}개')
    st.caption('코사인 유사도는 TF-IDF 가중치를 포함한 문서 방향의 유사도이고, 교집합 비율은 키워드 집합의 단순 겹침 비율입니다.')


def highlight_keyword_html(text: str, keyword: str, syn: Dict[str, str]) -> str:
    aliases = [keyword] + [source for source, target in syn.items() if target == keyword]
    aliases = sorted({alias for alias in aliases if alias}, key=len, reverse=True)
    if not aliases:
        return html.escape(str(text))
    pattern = re.compile('|'.join(re.escape(alias) for alias in aliases), flags=re.IGNORECASE)
    parts = []
    cursor = 0
    for match in pattern.finditer(str(text)):
        parts.append(html.escape(str(text)[cursor:match.start()]))
        parts.append(f'<mark class="sre-keyword-highlight">{html.escape(match.group(0))}</mark>')
        cursor = match.end()
    parts.append(html.escape(str(text)[cursor:]))
    return ''.join(parts)


def evidence_html(evd: pd.DataFrame, keyword: str, syn: Dict[str, str]) -> str:
    parts = ['<div class="sre-evidence-scroll">']
    parts.append(f'<div class="sre-selected-keyword">선택 키워드 · {html.escape(keyword)}</div>')
    for _, erow in evd.head(80).iterrows():
        meta = f"[{erow['출처']}] 문장 {erow['문장번호']} · {erow['등장횟수']}회"
        text = str(erow.get('강조원문', erow.get('원문', ''))).replace('**', '')
        highlighted = highlight_keyword_html(text, keyword, syn)
        parts.append(f'<div class="sre-evidence-card"><div class="sre-evidence-meta">{html.escape(meta)}</div><div class="sre-evidence-text">{highlighted}</div></div>')
    parts.append('</div>')
    return ''.join(parts)


def render_local_notice():
    body = '''
이 프로그램은 **교사의 PC에서 직접 실행하는 로컬 환경**을 전제로 설계되었습니다.

- 학생부 원문과 분석 요청은 커리어넷 API로 전송하지 않습니다.
- 커리어넷 갱신 기능은 **API 키와 공개 학과정보 요청값**만 커리어넷으로 보냅니다.
- 업로드한 파일은 실행 중 메모리와 사용자 PC의 임시 파일에 존재할 수 있습니다. 내장 DB 저장이나 다운로드를 선택하면 결과 파일이 PC에 남습니다.
- 앱을 외부 서버에 배포하면 더 이상 같은 의미의 로컬 처리가 아닙니다. 실제 학생 자료는 학교가 승인한 PC와 저장 위치에서만 처리하세요.
- 이 프로그램이 개인정보 보호 책임이나 학교 내부 승인 절차를 대신하지는 않습니다. 접근 권한, 보관 기간, 비식별화, 결과 공유 범위는 소속 기관 지침을 따라야 합니다.
'''
    render_help('개인정보·로컬 처리 안내', body)



def render_sidebar_privacy_notice():
    with st.sidebar.expander('🔒 로컬 분석 안내', expanded=False):
        st.markdown("""
이 프로그램은 **교사의 PC에서 실행하는 로컬 사용**을 전제로 합니다.

- 학생부 원문은 커리어넷 API로 전송하지 않습니다.
- 커리어넷 갱신 시에는 API 키와 공개 학과정보 요청만 외부로 나갑니다.
- 업로드·생성 파일은 실행 중 메모리 또는 사용자 PC의 임시·저장 파일로 남을 수 있습니다.
- 외부 서버에 이 앱을 배포하면 로컬 처리 조건이 달라집니다.

실제 학생 자료는 학교가 승인한 기기·계정·저장 위치에서 처리하고, 기관의 개인정보 보호 및 보관 지침을 준수하세요.
""")


def render_student_file_help():
    with st.expander('📁 나이스 학생부 파일 준비 방법', expanded=False):
        st.markdown("""
이 앱은 **나이스(NEIS)에서 내려받은 `.xlsx` 형식의 학생부 항목별 조회 파일**을 기준으로 작동합니다.  
아래 경로에서 각각의 파일을 내려받아 업로드해 주세요.

**1. 창체 파일(.xlsx)**  
`나이스 > 학급담임 > 학교생활기록부 > 학생부 항목별 조회 > 창의적체험활동 > 창의적체험활동`

**2. 교과세특 파일(.xlsx)**  
`나이스 > 학급담임 > 학교생활기록부 > 학생부 항목별 조회 > 교과학습발달상황 > 세부능력및특기사항`

**3. 행발 파일(.xlsx)**  
`나이스 > 학급담임 > 학교생활기록부 > 학생부 항목별 조회 > 행동특성및종합의견 > 행동특성및종합의견`

각 항목에서 조회한 뒤 **엑셀 저장/다운로드**를 선택하여 `.xlsx` 파일로 저장하면 됩니다.

학생 자료를 다룰 때에는 반드시 학교 내부 지침에 따라 접근 권한, 개인정보 보호, 비식별화 절차를 확인해 주세요.
""")


def render_first_use_guide():
    with st.expander('📘 처음 사용하는 교사를 위한 빠른 안내', expanded=False):
        st.markdown("""
### 이 프로그램은 무엇을 하나요?

학생부의 창체·교과세특·행발 서술을 단어와 문장 단위로 정리하고, 기록에서 상대적으로 두드러지는 표현을 찾습니다. 또한 커리어넷 학과 설명과 텍스트를 비교해 상담 때 다시 살펴볼 교과·활동·진로 키워드를 제시합니다.

### 권장 사용 순서

1. **학생부 전처리**: 원본 엑셀을 학생별로 병합하고 형태소·빈도·TF-IDF·근거문장 캐시를 만듭니다.
2. **말뭉치 준비**: 기존 학과 DB를 사용하거나 커리어넷 API로 공개 학과정보를 갱신합니다.
3. **학생 분석**: 학생을 선택하고 특징어, 원문 근거, 학과별 텍스트 연계도를 확인합니다.
4. **교사 확인**: 결과를 원문과 대조하고 교육적 맥락을 반영해 상담 참고자료로만 사용합니다.

### 꼭 알아둘 점

- 높은 유사도는 합격 가능성, 진로 적합성, 역량 수준을 뜻하지 않습니다.
- 낮은 유사도는 학생에게 역량이나 경험이 없다는 뜻이 아닙니다. 학생부에 해당 표현이 적게 기록되었을 수 있습니다.
- ‘보완 키워드’는 결핍 진단이 아니라 목표 학과 설명에는 있으나 현재 학생부 텍스트에서 상대적으로 덜 나타난 표현입니다.
- 결과를 학생 평가, 서열화, 자동 진로 배정의 근거로 단독 사용하지 마세요.
""")


def render_analysis_terms_help():
    render_help('분석 용어 안내', """
- **형태소 분석**: 문장을 의미 있는 단위로 나누는 과정입니다. 이 앱은 주로 명사와 영문 용어를 사용합니다.
- **Kiwi**: 한국어 문장을 더 정교하게 나누는 형태소 분석기입니다. 최초 전처리는 느리지만 조사·어미가 붙은 표현을 비교적 안정적으로 정리합니다.
- **간이 토큰화**: 글자 규칙과 간단한 어미 제거를 사용하는 빠른 방식입니다. 속도는 빠르지만 표현 잡음이 더 남을 수 있습니다.
- **TF-IDF**: 모든 학생에게 흔한 단어보다 특정 학생 기록에서 상대적으로 두드러지는 단어에 더 큰 값을 주는 통계값입니다.
- **코사인 유사도**: 두 문서의 단어 가중치 방향이 얼마나 비슷한지 나타냅니다. 확률이나 백분위가 아닙니다.
- **키워드 교집합 비율**: 두 문서의 주요 단어 집합이 단순히 얼마나 겹치는지 보여줍니다. TF-IDF 유사도와 계산 방식이 다릅니다.
- **말뭉치**: 비교 기준이 되는 학과 설명 텍스트 묶음입니다. 학업·교과·활동·적성·진로 관점으로 나누어 볼 수 있습니다.
- **적합·높은 연계 표시**: 현재 불러온 학생 전체와 학과 전체의 텍스트 유사도 분포에서 각각 상위 1%와 상위 5%에 해당하는 결과입니다. 진로 적성이나 합격 가능성을 뜻하지 않습니다.
- **보완 키워드**: 목표 학과 말뭉치에는 상대적으로 강하지만 학생부 기록에는 약한 표현입니다. 학생의 실제 결핍을 의미하지 않습니다.
- **근거 원문**: 추출된 키워드가 학생부의 어느 문장에서 나타났는지 확인하는 기능입니다. 최종 해석은 반드시 원문을 함께 읽고 판단하세요.
""")


def render_full_user_guide():
    st.subheader('처음 사용하는 교사를 위한 사용 안내')
    st.caption('프로그램의 사용 순서, 용어, 개인정보 보호, 결과 해석 및 문제 해결을 한곳에서 확인할 수 있습니다.')

    st.markdown("""
### 1. 프로그램의 목적과 한계

학생부 탐색기는 창체·교과세특·행발의 서술형 기록을 빠르게 탐색하도록 돕는 **교사용 보조 도구**입니다. 학생의 진로를 자동 결정하거나 역량을 평가하는 시스템이 아닙니다. 분석 결과는 원문 검토와 상담 질문을 시작하기 위한 단서로 사용하세요.

### 2. 처음 사용할 때

1. 나이스 학생부 항목별 조회에서 창체·교과세특·행발 `.xlsx` 파일을 준비합니다.
2. `학생부 전처리` 탭에 파일을 올리고 학년·반·번호·성명을 확인합니다.
3. Kiwi 또는 간이 토큰화를 선택해 전처리를 실행합니다.
4. 전처리 결과를 저장합니다. 이후에는 원본 대신 이 결과 파일을 불러오면 빠릅니다.
5. 학과 말뭉치를 불러오거나 커리어넷 API로 갱신한 후 내장 DB로 저장합니다.
6. `학생 분석`에서 학생과 분석 관점을 선택해 결과와 근거 원문을 함께 확인합니다.

### 3. 형태소 분석기 선택

형태소 분석기는 `학생부 전처리` 탭에서 Kiwi·MeCab·간이 토큰화 중 선택합니다. 현재 환경에 설치되지 않은 분석기는 실행할 수 없습니다. 이미 만든 학생부 캐시는 생성 당시 방식을 유지하므로 방식을 바꾸려면 원본 학생부를 다시 전처리해야 합니다. 학과 말뭉치 인덱스도 같은 방식으로 한 번 생성한 뒤 재사용합니다.

### 4. 학과 분석 관점

- **통합**: 학과 설명 전체를 종합적으로 비교합니다.
- **학업**: 학과 개요·특성·대학 주요 교과목 중심입니다.
- **교과**: 관련 고교 교과와 대학 교과목 중심입니다.
- **활동**: 진로 탐색 및 권장 활동 설명 중심입니다.
- **적성**: 커리어넷의 흥미와 적성 설명 중심입니다.
- **진로**: 졸업 후 진출 분야·관련 직업·자격 중심입니다.

관점별 점수는 서로 다른 텍스트를 기준으로 계산하므로 단순 비교하거나 평균을 학생 평가값으로 사용하지 마세요.

유사 학과 목록의 `적합` 표시는 고정 점수 기준이 아니라 현재 불러온 학생 집단과 전체 학과의 텍스트 비교값 중 상위 1%에 붙습니다. `높은 연계`는 상위 5%입니다. 집단·말뭉치·분석 설정이 달라지면 경계값도 달라지며, 이 표시는 학생의 진로 적성을 판정하지 않습니다.

### 5. 개인정보와 네트워크

- 로컬 실행에서는 학생부가 교사 PC에서 처리되며 커리어넷으로 전송되지 않습니다.
- 커리어넷 갱신은 API 키와 공개 학과정보 요청만 전송합니다.
- 업로드 파일과 임시 DB는 실행 중 메모리 또는 PC의 임시 파일에 남을 수 있습니다.
- 외부 서버나 공용 PC에서 실행하면 저장·전송 조건이 달라질 수 있으므로 실제 학생 자료를 사용하지 마세요.
- 결과 파일에도 학생 식별정보와 원문이 포함될 수 있습니다. 저장 위치, 접근 권한, 보관 기간과 삭제 절차를 학교 지침에 맞게 관리하세요.

### 6. 결과를 안전하게 해석하는 법

- 학생부는 학생의 모든 경험을 담은 완전한 자료가 아니라 교사가 기록한 문서입니다.
- 단어 빈도는 중요성이나 우수성을 직접 의미하지 않습니다.
- 유사도는 문장 의미 전체가 아니라 사용된 단어의 통계적 가까움을 중심으로 계산합니다.
- 학과 말뭉치의 내용과 갱신 시점에 따라 결과가 달라질 수 있습니다.
- 학생에게 결과를 제시할 때는 점수보다 공통 키워드와 근거 문장을 중심으로 대화하세요.
""")

    with st.expander('문제가 생겼을 때 확인할 사항', expanded=False):
        st.markdown("""
- **처음 분석이 오래 걸림**: Kiwi 학과 인덱스를 최초로 만드는 중일 수 있습니다. 같은 데이터와 설정에서는 캐시를 재사용합니다.
- **학생이 안 보임**: 원본 파일 형식과 학년·반·번호·성명 인식 결과를 병합 미리보기에서 확인하세요.
- **학생이 중복됨**: 서로 다른 파일의 학년·반 표기와 파일명을 확인하세요.
- **학과 결과가 없음**: 사이드바에 학과 말뭉치를 불러왔는지 또는 내장 DB가 있는지 확인하세요.
- **커리어넷 수집 실패**: API 키 승인·만료일, 인터넷 연결, 커리어넷 서비스 상태를 확인하고 오류·누락 학과만 다시 수집하세요.
- **이전과 점수가 다름**: 형태소 분석 방식, 불용어, 표현 통일 사전, 학생부 캐시, 학과 말뭉치 버전과 분석 관점이 같은지 확인하세요.
""")

    render_local_notice()
    render_analysis_terms_help()

def render_footer():
    st.divider()
    with st.container(border=True):
        product, author = st.columns([0.8, 1.2])
        with product:
            st.caption('STUDENT RECORD EXPLORER')
            st.markdown('#### 학생부 탐색기')
            st.caption('교육 현장을 위한 학생부 텍스트 분석 도구')
        with author:
            st.markdown('**박효진** · 개발 · 연구')
            st.caption('경남대학교 국어교육과 박사과정')
            st.caption('거제상문고등학교 교사')
            st.caption('losaci@naver.com')

        st.caption(
            '본 프로그램은 학교생활기록부 서술형 기록의 교육적 탐색을 지원하기 위한 연구·교육용 도구입니다. '
            '실제 학생 자료를 사용할 때에는 개인정보 보호, 비식별화, 기관 지침 및 연구윤리 절차를 준수해야 합니다. '
            '분석 결과는 진로 결정이나 평가의 최종 판단이 아니라 교사의 기록 탐색과 상담을 돕는 참고자료입니다.'
        )
        st.caption(
            f'Student Record Explorer v{APP_VERSION} · For educational & research use · '
            'Copyright © Park Hyojin. All rights reserved.'
        )

def is_student_row(vals):
    return len(vals)>=2 and NO_RE.match(clean(vals[0]) or '') and NAME_RE.match(clean(vals[1]) or '')


def header_footer(vals):
    j=' '.join(clean(x) for x in vals if clean(x))
    return (not j) or any(k in j for k in ['번 호','성 명','학교생활기록부','사용자명','학년','/'])


def row_text(vals, source, has_cols):
    start=2 if has_cols else 0; cand=[]
    for v in [clean(x) for x in vals[start:]]:
        if not v or header_footer([v]) or NO_RE.match(v): continue
        if source=='창체' and v in {'자율활동','동아리활동','봉사활동','진로활동','영역','시간','특기사항'}: continue
        if len(v)<6 and '희망' not in v: continue
        cand.append(v)
    return ' '.join(cand)


def parse_excel(file, source):
    raw=pd.read_excel(file, header=None, dtype=str, engine='openpyxl').fillna('')
    file_name=getattr(file, 'name', '')
    grade,class_no=extract_grade_class(raw, file_name)
    records={}; current=None; pieces=0
    for row in raw.values.tolist():
        vals=[clean(x) for x in row]
        if is_student_row(vals):
            no,name=normalize_school_number(vals[0]),vals[1]
            current=(grade,class_no,no,name)
            records.setdefault(current, {'학년':grade, '반':class_no, '번호':no, '성명':name, source:''})
            tx=row_text(vals, source, True)
            if tx: records[current][source]+=' '+tx; pieces+=1
        elif current and not header_footer(vals):
            tx=row_text(vals, source, False)
            if tx: records[current][source]+=' '+tx; pieces+=1
    df=pd.DataFrame(records.values()) if records else pd.DataFrame(columns=STUDENT_ID_COLUMNS+[source])
    if not df.empty:
        df[source]=df[source].fillna('').map(clean)
        df=df[df[source].str.len()>0].reset_index(drop=True)
    group_label = f'{grade}학년 {class_no}반' if grade and class_no else '학년·반 미확인'
    return df, f'{source}: {group_label}, {len(df)}명, 조각 {pieces}개'


def parse_excel_files(
    files,
    source,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """같은 항목의 엑셀 파일을 여러 개 받아 하나로 합칩니다.

    예: 3-1반 창체.xlsx, 3-2반 창체.xlsx를 함께 넣으면
    두 파일의 학생 기록을 하나의 창체 자료로 통합합니다.
    """
    if not files:
        return None, ''
    parsed=[]; messages=[]; unidentified=[]
    total_pieces=0
    file_count = len(files)
    for file_index, f in enumerate(files, start=1):
        if progress_callback:
            progress_callback(
                (file_index - 1) / file_count,
                f'{source} 파일 읽는 중 · {file_index}/{file_count} · {getattr(f, "name", "파일")}',
            )
        df, msg = parse_excel(f, source)
        parsed.append(df)
        messages.append(f"{getattr(f, 'name', '파일')}: {msg}")
        if not df.empty and ((df['학년'] == '').any() or (df['반'] == '').any()):
            unidentified.append(getattr(f, 'name', '파일'))
        try:
            total_pieces += int(msg.split('조각 ')[1].split('개')[0])
        except Exception:
            pass
        if progress_callback:
            progress_callback(
                file_index / file_count,
                f'{source} 파일 읽기 완료 · {file_index}/{file_count}',
            )
    parsed=[d for d in parsed if d is not None and not d.empty]
    if not parsed:
        return pd.DataFrame(columns=STUDENT_ID_COLUMNS+[source]), f'{source}: 0명, 조각 0개'
    df_all=pd.concat(parsed, ignore_index=True)
    # 같은 학생이 여러 파일에 나뉘어 들어간 경우 같은 항목 텍스트를 합칩니다.
    df_all[source]=df_all[source].fillna('').astype(str)
    df_all=(df_all.groupby(STUDENT_ID_COLUMNS, as_index=False, dropna=False)[source]
            .agg(lambda x: clean(' '.join([str(v) for v in x if str(v).strip()]))))
    warning = f', 학년·반 미확인 {len(unidentified)}개' if unidentified else ''
    return df_all, f'{source}: 파일 {len(files)}개, 학생 {len(df_all)}명, 조각 {total_pieces}개{warning}'


def merge_records(dfs):
    base=None
    for df in dfs:
        if df is None or df.empty: continue
        df=ensure_student_identity_columns(df)
        base=df.copy() if base is None else pd.merge(base, df, on=STUDENT_ID_COLUMNS, how='outer')
    if base is None: return pd.DataFrame(columns=STUDENT_ID_COLUMNS+['창체','교과세특','행발','통합'])
    for c in ['창체','교과세특','행발']:
        if c not in base.columns: base[c]=''
        base[c]=base[c].fillna('').astype(str)
    base['통합']=(base['창체']+' '+base['교과세특']+' '+base['행발']).map(clean)
    base['_g']=pd.to_numeric(base['학년'], errors='coerce')
    base['_c']=pd.to_numeric(base['반'], errors='coerce')
    base['_n']=pd.to_numeric(base['번호'], errors='coerce')
    return (base.sort_values(['_g','_c','_n','성명'], na_position='last')
            .drop(columns=['_g','_c','_n']).reset_index(drop=True))


def font_path():
    candidates = []
    if platform.system().lower().startswith('win'):
        candidates += [r'C:\Windows\Fonts\malgun.ttf']
    candidates += ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    for c in candidates:
        if Path(c).exists(): return c
    return None


def wordcloud_fig(freq: Dict[str,float]):
    if not freq: return None
    wc=WordCloud(font_path=font_path(), width=1400, height=760, background_color='white', collocations=False, max_words=120, random_state=42, margin=8).generate_from_frequencies(freq)
    fig,ax=plt.subplots(figsize=(10.5,5.7)); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); fig.tight_layout(pad=0); return fig


def tfidf_table(df, col, stop, syn, min_len, use_kiwi):
    docs=[tokenized(x, stop, syn, min_len, use_kiwi) for x in df[col].fillna('').astype(str)]
    if not any(docs): return pd.DataFrame(), None, None
    vec=TfidfVectorizer(token_pattern=r'(?u)\b\w+\b'); mat=vec.fit_transform(docs); terms=vec.get_feature_names_out(); rows=[]
    for i,row in df.iterrows():
        scores=mat[i].toarray().ravel()
        for rank,idx in enumerate(scores.argsort()[::-1][:30],1):
            if scores[idx] > 0:
                identity = {column: row.get(column, '') for column in STUDENT_ID_COLUMNS}
                rows.append({**identity, '순위':rank, '단어':terms[idx], 'TF-IDF':round(float(scores[idx]),4)})
    return pd.DataFrame(rows), vec, mat


def save_db(df, path=DB_PATH):
    DATA_DIR.mkdir(exist_ok=True)
    df = enrich_existing_corpus(df)
    with sqlite3.connect(path) as conn:
        df.fillna('').to_sql('majors', conn, if_exists='replace', index=False)


def load_db(path=DB_PATH):
    if not Path(path).exists(): return pd.DataFrame()
    try:
        with sqlite3.connect(path) as conn: return pd.read_sql('select * from majors', conn).fillna('')
    except Exception: return pd.DataFrame()


def read_corpus(uploaded):
    if uploaded is None: return pd.DataFrame()
    name=uploaded.name.lower()
    if name.endswith(('.db','.sqlite','.sqlite3')):
        tmp=DATA_DIR/'_tmp_corpus.db'; DATA_DIR.mkdir(exist_ok=True); tmp.write_bytes(uploaded.getvalue()); result = load_db(tmp)
    elif name.endswith('.xlsx'):
        result = pd.read_excel(uploaded, dtype=str, engine='openpyxl').fillna('')
    else:
        result = pd.read_csv(uploaded, dtype=str).fillna('')
    return enrich_existing_corpus(result)


def corpus_texts(df, channel: str = '통합'):
    channel_column = f'말뭉치_{channel}'
    if channel_column in df.columns: return df[channel_column].astype(str).tolist()
    if '말뭉치' in df.columns: return df['말뭉치'].astype(str).tolist()
    cols=[c for c in df.columns if c not in {'majorSeq','계열','학과명','학과'}]
    return df[cols].astype(str).agg(' '.join, axis=1).tolist()


def _major_index(major_df, channel, stop, syn, min_len, use_kiwi):
    documents = tuple(corpus_texts(major_df, channel))
    return prepare_major_index(
        documents,
        tuple(sorted(stop)),
        tuple(sorted(syn.items())),
        min_len,
        use_kiwi,
    )


def similarity(student_text, major_df, stop, syn, min_len, use_kiwi, top_k=10, channel='통합'):
    if major_df.empty or not student_text.strip(): return pd.DataFrame()
    vec, mat, terms = _major_index(major_df, channel, stop, syn, min_len, use_kiwi)
    if vec is None: return pd.DataFrame()
    student_doc = tokenized(student_text, stop, syn, min_len, use_kiwi)
    if not student_doc: return pd.DataFrame()
    student_vector = vec.transform([student_doc])
    sims=cosine_similarity(student_vector, mat).ravel()
    s_scores=student_vector.toarray().ravel(); s_top={terms[i] for i in s_scores.argsort()[::-1][:50] if s_scores[i]>0}
    rows=[]; name_col='학과명' if '학과명' in major_df.columns else ('학과' if '학과' in major_df.columns else major_df.columns[0])
    for rank,idx in enumerate(sims.argsort()[::-1][:top_k],1):
        m_scores=mat[idx].toarray().ravel(); m_top={terms[i] for i in m_scores.argsort()[::-1][:50] if m_scores[i]>0}
        rows.append({'순위':rank,'학과명':major_df.iloc[idx].get(name_col,''),'계열':major_df.iloc[idx].get('계열',''),'유사도':round(float(sims[idx]),4),'공통핵심어':', '.join(list(s_top & m_top)[:12])})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, persist='disk', max_entries=24)
def calibrate_similarity_thresholds(
    student_documents: Tuple[str, ...],
    corpus_documents: Tuple[str, ...],
    stop_words: Tuple[str, ...],
    synonym_items: Tuple[Tuple[str, str], ...],
    min_len: int,
    analyzer: Any,
) -> Dict[str, float]:
    """현재 학생 집단 × 전체 학과의 경험적 유사도 분포로 표시 기준을 정합니다."""
    stop = set(stop_words)
    syn = dict(synonym_items)
    vec, major_matrix, _ = prepare_major_index(
        corpus_documents, stop_words, synonym_items, min_len, analyzer
    )
    if vec is None or major_matrix is None:
        return {'fit': 1.0, 'high': 1.0, 'medium': 1.0, 'comparisons': 0}
    token_docs = [tokenized(text, stop, syn, min_len, analyzer) for text in student_documents]
    token_docs = [text for text in token_docs if text]
    if not token_docs:
        return {'fit': 1.0, 'high': 1.0, 'medium': 1.0, 'comparisons': 0}
    student_matrix = vec.transform(token_docs)
    scores = pd.Series(cosine_similarity(student_matrix, major_matrix).ravel())
    if scores.empty:
        return {'fit': 1.0, 'high': 1.0, 'medium': 1.0, 'comparisons': 0}
    fit = max(float(scores.quantile(0.99)), 0.01)
    high = min(fit, max(float(scores.quantile(0.95)), 0.005))
    medium = min(high, max(float(scores.quantile(0.75)), 0.001))
    return {
        'fit': fit,
        'high': high,
        'medium': medium,
        'comparisons': int(len(scores)),
        'students': int(len(token_docs)),
        'majors': int(major_matrix.shape[0]),
    }


def label_similarity_results(sim: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    result = sim.copy()
    result['연계수준'] = result['유사도'].astype(float).map(
        lambda score: similarity_level(score, thresholds)
    )
    result['유사도(%)'] = result['유사도'].astype(float) * 100
    return result[['순위', '학과명', '계열', '유사도(%)', '연계수준', '공통핵심어']]


def style_similarity_results(df: pd.DataFrame):
    palette = {
        '적합': 'background-color: rgba(34,197,94,.24); color: #DCFCE7; font-weight: 800;',
        '높은 연계': 'background-color: rgba(59,130,246,.22); color: #DBEAFE; font-weight: 750;',
        '보통 연계': 'background-color: rgba(245,158,11,.18); color: #FEF3C7;',
        '낮은 연계': 'background-color: rgba(148,163,184,.10); color: #CBD5E1;',
    }

    def color_row(row):
        style = palette.get(row.get('연계수준'), '')
        return [style] * len(row)

    return df.style.apply(color_row, axis=1).format({'유사도(%)': '{:.1f}%'})


def gap_analysis(student_text, target_row, major_df, stop, syn, min_len, use_kiwi, top_n=30, channel='통합'):
    name_col='학과명' if '학과명' in target_row.index else ('학과' if '학과' in target_row.index else '')
    channel_column = f'말뭉치_{channel}'
    target_text = str(target_row.get(channel_column, target_row.get('말뭉치','')))
    vec, _, terms = _major_index(major_df, channel, stop, syn, min_len, use_kiwi)
    student_doc = tokenized(student_text, stop, syn, min_len, use_kiwi)
    target_doc = tokenized(target_text, stop, syn, min_len, use_kiwi)
    if vec is None or not student_doc or not target_doc: return pd.DataFrame(), pd.DataFrame(), {}, {}
    pair = vec.transform([student_doc, target_doc])
    sv=pair[0].toarray().ravel(); tv=pair[1].toarray().ravel(); rows=[]; common=[]
    for i,t in enumerate(terms):
        if tv[i]-sv[i] > 0: rows.append({'키워드':t,'목표학과_TFIDF':round(float(tv[i]),4),'학생부_TFIDF':round(float(sv[i]),4),'부족도':round(float(tv[i]-sv[i]),4)})
        if tv[i]>0 and sv[i]>0: common.append({'키워드':t,'공통점수':round(float(min(tv[i],sv[i])),4)})
    gap=pd.DataFrame(rows).sort_values('부족도', ascending=False).head(top_n) if rows else pd.DataFrame()
    com=pd.DataFrame(common).sort_values('공통점수', ascending=False).head(top_n) if common else pd.DataFrame()
    s_terms={terms[i] for i in sv.argsort()[::-1][:200] if sv[i]>0}; t_terms={terms[i] for i in tv.argsort()[::-1][:200] if tv[i]>0}; inter=s_terms&t_terms; union=s_terms|t_terms
    metrics={'student_count':len(s_terms),'target_count':len(t_terms),'common_count':len(inter),'jaccard':len(inter)/len(union) if union else 0,'cosine':float(cosine_similarity(pair[0],pair[1]).ravel()[0]), 'student_only':s_terms-inter,'target_only':t_terms-inter,'common_terms':inter}
    return gap, com, dict(zip(gap['키워드'], gap['부족도'])) if not gap.empty else {}, metrics


def venn_fig(metrics):
    fp=font_path(); prop=fm.FontProperties(fname=fp) if fp else None; kw={'fontproperties':prop} if prop else {}
    fig,ax=plt.subplots(figsize=(5.4,3.05), dpi=160); fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#0e1117')
    sc,tc,cc=metrics.get('student_count',0),metrics.get('target_count',0),metrics.get('common_count',0); mx=max(sc,tc,1)
    ax.add_patch(Circle((.40,.55), .245+.055*sc/mx, color='#3B82F6', alpha=.56)); ax.add_patch(Circle((.62,.55), .245+.055*tc/mx, color='#22C55E', alpha=.56))
    ax.add_patch(FancyBboxPatch((.445,.405),.13,.25,boxstyle='round,pad=.012,rounding_size=.025',linewidth=0,facecolor='#14B8A6',alpha=.34))
    ax.text(.25,.81,'학생부',color='#DBEAFE',ha='center',fontsize=10.5,fontweight='bold',**kw); ax.text(.77,.81,'목표 학과',color='#DCFCE7',ha='center',fontsize=10.5,fontweight='bold',**kw)
    ax.text(.28,.55,f'학생부만\n{max(sc-cc,0):,}개',color='white',ha='center',va='center',fontsize=10.5,**kw)
    ax.text(.51,.55,f'공통\n{cc:,}개\n{metrics.get("jaccard",0)*100:.1f}%',color='white',ha='center',va='center',fontsize=12.5,fontweight='bold',**kw)
    ax.text(.75,.55,f'학과만\n{max(tc-cc,0):,}개',color='white',ha='center',va='center',fontsize=10.5,**kw)
    ax.text(.51,.18,f'TF-IDF 코사인 유사도  {metrics.get("cosine",0)*100:.1f}%',color='#E5E7EB',ha='center',fontsize=11.5,fontweight='bold',**kw)
    ax.text(.51,.105,'집합 비율과 코사인 유사도는 서로 다른 지표입니다.',color='#9CA3AF',ha='center',fontsize=8.5,**kw)
    ax.set_xlim(.03,.99); ax.set_ylim(.02,.98); ax.axis('off'); fig.tight_layout(pad=.25); return fig


def df_bytes(df, kind):
    if kind=='CSV': return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'), 'text/csv', 'major_corpus.csv'
    out=io.BytesIO(); df.to_excel(out,index=False); return out.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'major_corpus.xlsx'


def build_student_cache(
    merged: pd.DataFrame,
    scope: str,
    stop: set[str],
    syn: dict[str, str],
    min_len: int,
    use_kiwi: bool,
    top_n: int = 50,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """학생부 원자료를 한 번만 전처리하여 분석용 표 묶음을 생성합니다.

    records: 학생별 병합 원문
    tfidf: 학생별 TF-IDF 특징어
    freq: 학생별 빈도 상위어
    evidence: 출처/문장 단위 근거 인덱스
    meta: 저장 시 별도 JSON/테이블로 사용
    """
    if progress_callback:
        progress_callback(0.03, '학생부 자료를 정리하는 중입니다.')
    records = ensure_student_identity_columns(merged).fillna('')
    if progress_callback:
        progress_callback(0.10, '전체 학생의 TF-IDF 특징어를 계산하는 중입니다.')
    tfidf_df, _, _ = tfidf_table(records, scope, stop, syn, min_len, use_kiwi)
    if progress_callback:
        progress_callback(0.35, '학생별 빈도와 근거 문장을 만드는 중입니다.')

    freq_rows = []
    evidence_rows = []
    student_count = len(records)
    update_every = max(1, student_count // 100)
    for student_index, (_, row) in enumerate(records.iterrows(), start=1):
        name = str(row.get('성명', ''))
        no = str(row.get('번호', ''))
        identity = {column: str(row.get(column, '')) for column in STUDENT_ID_COLUMNS}
        text = str(row.get(scope, ''))
        toks = tokenize(text, stop, syn, min_len, use_kiwi)
        for rank, (word, cnt) in enumerate(Counter(toks).most_common(top_n), 1):
            freq_rows.append({**identity, '분석범위': scope, '순위': rank, '단어': word, '빈도': int(cnt)})

        for source in ['창체', '교과세특', '행발']:
            sentences = split_record_sentences(str(row.get(source, '')))
            for i, sent in enumerate(sentences, 1):
                sent_tokens = tokenize(sent, stop, syn, min_len, use_kiwi)
                if not sent_tokens:
                    continue
                uniq = sorted(set(sent_tokens))
                evidence_rows.append({
                    **identity,
                    '출처': source,
                    '문장번호': i,
                    '원문': sent,
                    '키워드목록': ', '.join(uniq),
                })
        if progress_callback and (student_index % update_every == 0 or student_index == student_count):
            progress_callback(
                0.35 + (0.60 * student_index / max(student_count, 1)),
                f'학생별 자료 처리 중 · {student_index}/{student_count}명',
            )

    if progress_callback:
        progress_callback(0.97, '전처리 결과를 정리하는 중입니다.')
    freq_df = pd.DataFrame(freq_rows)
    evidence_df = pd.DataFrame(evidence_rows)
    meta_df = pd.DataFrame([{
        '분석범위': scope,
        '학생수': len(records),
        '생성시각': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        '최소단어길이': min_len,
        '형태소분석기': analyzer_name(use_kiwi),
    }])
    result = {'records': records, 'tfidf': tfidf_df, 'freq': freq_df, 'evidence': evidence_df, 'meta': meta_df}
    if progress_callback:
        progress_callback(1.0, '전처리가 완료되었습니다.')
    return result


def normalize_student_cache_identity(cache: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    for key in ['records', 'tfidf', 'freq', 'evidence']:
        if key in cache and isinstance(cache[key], pd.DataFrame):
            cache[key] = ensure_student_identity_columns(cache[key])
    return cache


def cache_to_zip(cache: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, df in cache.items():
            if isinstance(df, pd.DataFrame):
                zf.writestr(f'{name}.csv', df.to_csv(index=False, encoding='utf-8-sig'))
    return buf.getvalue()


def cache_to_excel(cache: Dict[str, pd.DataFrame]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        sheet_names = {'records':'학생부병합', 'tfidf':'학생별TFIDF', 'freq':'학생별빈도', 'evidence':'근거문장인덱스', 'meta':'분석정보'}
        for key, df in cache.items():
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=sheet_names.get(key, key)[:31], index=False)
    return out.getvalue()


def cache_to_db(cache: Dict[str, pd.DataFrame]) -> bytes:
    tmp = DATA_DIR / '_student_cache_download.db'
    DATA_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(tmp) as conn:
        for key, df in cache.items():
            if isinstance(df, pd.DataFrame):
                df.fillna('').to_sql(key, conn, if_exists='replace', index=False)
    return tmp.read_bytes()


def load_student_cache(uploaded) -> Optional[Dict[str, pd.DataFrame]]:
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith('.zip'):
            result = {}
            with zipfile.ZipFile(io.BytesIO(uploaded.getvalue())) as zf:
                for key in ['records', 'tfidf', 'freq', 'evidence', 'meta']:
                    fn = f'{key}.csv'
                    if fn in zf.namelist():
                        result[key] = pd.read_csv(zf.open(fn), dtype=str).fillna('')
            return normalize_student_cache_identity(result) if result else None
        if name.endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(uploaded)
            mapping = {'학생부병합':'records', '학생별TFIDF':'tfidf', '학생별빈도':'freq', '근거문장인덱스':'evidence', '분석정보':'meta'}
            result = {}
            for sh in xls.sheet_names:
                key = mapping.get(sh, sh)
                if key in {'records','tfidf','freq','evidence','meta'}:
                    result[key] = pd.read_excel(uploaded, sheet_name=sh, dtype=str).fillna('')
            return normalize_student_cache_identity(result) if result else None
        if name.endswith(('.db', '.sqlite', '.sqlite3')):
            tmp = DATA_DIR / '_uploaded_student_cache.db'
            DATA_DIR.mkdir(exist_ok=True)
            tmp.write_bytes(uploaded.getvalue())
            result = {}
            with sqlite3.connect(tmp) as conn:
                for key in ['records', 'tfidf', 'freq', 'evidence', 'meta']:
                    try:
                        result[key] = pd.read_sql(f'select * from {key}', conn).fillna('')
                    except Exception:
                        pass
            return normalize_student_cache_identity(result) if result else None
    except Exception as e:
        st.error(f'전처리 결과 파일을 읽는 중 오류가 발생했습니다: {e}')
        return None
    return None


def save_student_cache_db(cache: Dict[str, pd.DataFrame], path: Path = STUDENT_CACHE_PATH) -> None:
    if not isinstance(cache, dict):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for key in ['records', 'tfidf', 'freq', 'evidence', 'meta']:
            df = cache.get(key)
            if isinstance(df, pd.DataFrame):
                df.fillna('').to_sql(key, conn, if_exists='replace', index=False)


def load_student_cache_db(path: Path = STUDENT_CACHE_PATH) -> Optional[Dict[str, pd.DataFrame]]:
    if not path.exists():
        return None
    result: Dict[str, pd.DataFrame] = {}
    with sqlite3.connect(path) as conn:
        for key in ['records', 'tfidf', 'freq', 'evidence', 'meta']:
            try:
                result[key] = pd.read_sql(f'select * from {key}', conn).fillna('')
            except Exception:
                pass
    return normalize_student_cache_identity(result) if result else None


def parse_uploaded_records(
    f_ch,
    f_se,
    f_ha,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    dfs=[]; msgs=[]
    groups = [(f_ch,'창체'),(f_se,'교과세특'),(f_ha,'행발')]
    total_files = sum(len(files) for files, _ in groups if files)
    completed_files = 0
    for files,src in groups:
        if files:
            group_size = len(files)

            def report_group_progress(fraction: float, message: str) -> None:
                if progress_callback:
                    overall = (completed_files + fraction * group_size) / max(total_files, 1)
                    progress_callback(overall * 0.90, message)

            df,msg=parse_excel_files(files,src,report_group_progress)
            if df is not None:
                dfs.append(df)
            if msg:
                msgs.append(msg)
            completed_files += group_size
    if progress_callback:
        progress_callback(0.94, '학생별 자료를 하나로 병합하는 중입니다.')
    merged = merge_records(dfs)
    if progress_callback:
        progress_callback(1.0, f'파일 읽기와 병합이 완료되었습니다 · 학생 {len(merged)}명')
    return merged, msgs


def uploaded_records_signature(f_ch, f_se, f_ha) -> Tuple[Tuple[str, str, int], ...]:
    """업로드 구성이 바뀌었을 때만 원본 엑셀을 다시 읽도록 식별값을 만듭니다."""
    signature = []
    for files, source in [(f_ch, '창체'), (f_se, '교과세특'), (f_ha, '행발')]:
        for uploaded in files or []:
            signature.append((source, uploaded.name, int(getattr(uploaded, 'size', 0))))
    return tuple(signature)


def uploaded_file_signature(uploaded) -> Optional[Tuple[str, int]]:
    if uploaded is None:
        return None
    return uploaded.name, int(getattr(uploaded, 'size', 0))


def student_cache_analyzer(cache: Dict[str, pd.DataFrame]) -> Optional[str]:
    meta = cache.get('meta') if isinstance(cache, dict) else None
    if not isinstance(meta, pd.DataFrame) or meta.empty or '형태소분석기' not in meta.columns:
        return None
    mode = str(meta.iloc[0].get('형태소분석기', '')).strip()
    return analyzer_name(mode) if mode else None


def get_active_cache(merged: pd.DataFrame, scope: str, stop: set[str], syn: dict[str, str], min_len: int, use_kiwi: bool, top_n: int) -> Dict[str, pd.DataFrame]:
    cache = st.session_state.get('student_cache')
    if isinstance(cache, dict) and 'records' in cache and 'tfidf' in cache and 'freq' in cache:
        return cache
    if merged is not None and not merged.empty:
        return build_student_cache(merged, scope, stop, syn, min_len, use_kiwi, top_n)
    return {}


def initialize_session_from_disk() -> None:
    if st.session_state.get('_startup_restore_done'):
        return
    with st.status('저장된 데이터 확인 중입니다.', expanded=False) as startup_status:
        startup_progress = st.progress(0, text='환경 설정을 확인하는 중입니다.')
        startup_progress.progress(0.20, text='말뭉치 저장본을 확인하는 중입니다.')
        loaded_corpus = False
        if not isinstance(st.session_state.get('major_corpus_df'), pd.DataFrame) and DB_PATH.exists():
            try:
                major_df = load_db()
                if not major_df.empty:
                    st.session_state['major_corpus_df'] = major_df
                    st.session_state['major_corpus_name'] = f'자동 불러오기: {DB_PATH.name}'
                    loaded_corpus = True
            except Exception:
                pass

        startup_progress.progress(0.55, text='전처리 캐시 저장본을 확인하는 중입니다.')
        loaded_cache = False
        if not isinstance(st.session_state.get('student_cache'), dict):
            try:
                cached = load_student_cache_db()
                if isinstance(cached, dict) and not cached.get('records', pd.DataFrame()).empty:
                    st.session_state['student_cache'] = cached
                    st.session_state['student_cache_name'] = f'자동 불러오기: {STUDENT_CACHE_PATH.name}'
                    loaded_cache = True
            except Exception:
                pass

        startup_progress.progress(1.0, text='초기 확인이 완료되었습니다.')
        if loaded_corpus or loaded_cache:
            startup_status.update(label='저장된 데이터를 자동으로 불러왔습니다.', state='complete', expanded=False)
        else:
            startup_status.update(label='자동 불러올 저장 데이터가 없어 기본 상태로 시작합니다.', state='complete', expanded=False)
    st.session_state['_startup_restore_done'] = True


def main():
    ensure_files()
    st.set_page_config('StudentRecord Explorer', layout='wide')
    if 'theme_mode' not in st.session_state:
        st.session_state['theme_mode'] = 'dark'
    apply_ui_style()
    apply_theme_mode(st.session_state.get('theme_mode', 'dark'))
    initialize_session_from_disk()
    st.title('학생부 탐색기')
    st.caption('학생부의 정성적 서술 영역을 텍스트 마이닝으로 구조화·정량화하여, 교사의 신속하고 근거 있는 학생부 탐색을 돕습니다.')
    render_first_use_guide()
    render_help('결과 해석 안내', """
이 프로그램의 결과는 학생의 진로·역량·합격 가능성을 판정하는 값이 아닙니다. 창체·교과세특·행발에 **기록된 텍스트**와 공개 학과 설명에 나타난 단어의 상대적 특징과 가까움을 보여줍니다.

- 점수만 보지 말고 공통 키워드와 학생부 근거 원문을 함께 확인하세요.
- 낮은 점수나 보완 키워드를 학생의 부족함으로 해석하지 마세요.
- 결과는 상담 질문과 기록 검토를 돕는 참고자료로만 사용하세요.
""")

    # ------------------------------------------------------------------
    # Sidebar: 배포판 기준. 원본 학생부가 아니라 전처리 캐시와 학과 말뭉치를 우선 불러옵니다.
    # ------------------------------------------------------------------
    st.sidebar.header('1. 데이터 불러오기')
    render_sidebar_privacy_notice()
    st.sidebar.selectbox(
        '화면 테마',
        ['dark', 'light'],
        key='theme_mode',
        format_func=lambda value: '다크' if value == 'dark' else '라이트',
        help='기본은 다크 모드입니다. 변경 시 화면이 바로 새로고침됩니다.',
    )

    cache_file_sidebar = st.sidebar.file_uploader(
        '전처리된 학생부 파일(.zip/.xlsx/.db)',
        type=['zip', 'xlsx', 'db', 'sqlite', 'sqlite3'],
        key='sidebar_student_cache_upload',
    )
    if cache_file_sidebar is not None:
        cache_signature = uploaded_file_signature(cache_file_sidebar)
        if cache_signature != st.session_state.get('student_cache_upload_signature'):
            cache = load_student_cache(cache_file_sidebar)
            if cache:
                st.session_state['student_cache'] = cache
                st.session_state['student_cache_name'] = cache_file_sidebar.name
                st.session_state['student_cache_upload_signature'] = cache_signature
                save_student_cache_db(cache)

    corpus_file = st.sidebar.file_uploader(
        '학과 말뭉치(.db/.csv/.xlsx)',
        type=['db', 'sqlite', 'sqlite3', 'csv', 'xlsx'],
        key='sidebar_corpus_upload',
    )
    if corpus_file is not None:
        corpus_signature = uploaded_file_signature(corpus_file)
        if corpus_signature != st.session_state.get('major_corpus_upload_signature'):
            try:
                major_df_sidebar = read_corpus(corpus_file)
                if not major_df_sidebar.empty:
                    st.session_state['major_corpus_df'] = major_df_sidebar
                    st.session_state['major_corpus_name'] = corpus_file.name
                    st.session_state['major_corpus_upload_signature'] = corpus_signature
                    save_db(major_df_sidebar)
            except Exception as e:
                st.sidebar.error(f'말뭉치 파일을 읽지 못했습니다: {e}')

    cache_now = st.session_state.get('student_cache')
    if isinstance(cache_now, dict) and not cache_now.get('records', pd.DataFrame()).empty:
        st.sidebar.success(f"학생부 캐시: {len(cache_now.get('records', pd.DataFrame()))}명")
        st.sidebar.caption(st.session_state.get('student_cache_name', '현재 세션 캐시'))
    else:
        st.sidebar.info('학생부 캐시 없음')

    major_now = st.session_state.get('major_corpus_df')
    if isinstance(major_now, pd.DataFrame) and not major_now.empty:
        st.sidebar.success(f'학과 말뭉치: {len(major_now)}개')
        st.sidebar.caption(st.session_state.get('major_corpus_name', '현재 세션 말뭉치'))
    elif DB_PATH.exists():
        dbdf = load_db()
        st.sidebar.success(f'내장 학과 DB: {len(dbdf)}개')
    else:
        st.sidebar.info('학과 말뭉치 없음')

    st.sidebar.header('2. 분석 설정')
    scope = st.sidebar.radio('분석 범위', ['통합', '창체', '교과세특', '행발'])
    min_len = st.sidebar.slider('최소 단어 길이', 1, 4, 2)
    top_n = st.sidebar.slider('상위 단어 수', 10, 100, 50, step=5)
    extra = st.sidebar.text_area('추가 불용어')
    stop = read_stopwords(extra)
    syn = read_synonyms()
    st.sidebar.caption(f'불용어 {len(stop)}개 / 표현통일 {len(syn)}개')
    render_desktop_controls()
    requested_analyzer = st.session_state.get('preprocess_analyzer', 'Kiwi')

    # 탭 순서: 가장 많이 쓰는 학생 분석을 첫 번째에 둡니다.
    tab_analysis, tab_pre, tab_corpus, tab_dict, tab_help = st.tabs([
        '1. 학생 분석', '2. 학생부 전처리', '3. 말뭉치 관리', '4. 분석 사전', '5. 사용 안내'
    ])

    # ------------------------------------------------------------------
    # 1. 학생 분석: 전처리된 캐시를 바로 읽어 분석합니다.
    # ------------------------------------------------------------------
    with tab_analysis:
        render_analysis_terms_help()
        cache = st.session_state.get('student_cache')
        if not isinstance(cache, dict) or cache.get('records', pd.DataFrame()).empty:
            st.info('왼쪽 사이드바에서 전처리된 학생부 파일을 불러오세요. 파일이 없다면 2번 「학생부 전처리」 탭에서 원본 학생부 파일을 넣고 먼저 전처리하면 됩니다.')
            render_footer()
        else:
            cached_analyzer = student_cache_analyzer(cache)
            effective_analyzer = cached_analyzer or requested_analyzer
            if cached_analyzer is not None and cached_analyzer != analyzer_name(requested_analyzer):
                st.info(
                    f'불러온 학생부 캐시는 **{cached_analyzer}** 방식으로 전처리되었습니다. '
                    '분석 일관성을 위해 이 캐시를 사용하는 동안에는 같은 방식을 적용합니다. '
                    '방식을 바꾸려면 원본 학생부를 다시 전처리해 주세요.'
                )
            records = ensure_student_identity_columns(cache.get('records', pd.DataFrame())).fillna('')
            tfidf_df = ensure_student_identity_columns(cache.get('tfidf', pd.DataFrame())).fillna('')
            freq_df = ensure_student_identity_columns(cache.get('freq', pd.DataFrame())).fillna('')
            evidence_df = ensure_student_identity_columns(cache.get('evidence', pd.DataFrame())).fillna('')

            render_status_card(
                len(records),
                scope,
                int((records['창체'].astype(str).str.len() > 0).sum()) if '창체' in records.columns else 0,
                int((records['교과세특'].astype(str).str.len() > 0).sum()) if '교과세특' in records.columns else 0,
                int((records['행발'].astype(str).str.len() > 0).sum()) if '행발' in records.columns else 0,
            )

            l, r = st.columns([0.95, 2.05])
            with l:
                card_open('학생 선택', '분석할 학생과 원문 확인 범위를 선택합니다.')
                q = st.text_input('학생 검색', key='analysis_student_search')
                labels = {idx: student_label(row) for idx, row in records.iterrows()}
                student_options = [idx for idx, label in labels.items() if not q or q in label]
                if not student_options:
                    st.warning('검색 결과가 없습니다.')
                    render_footer()
                    return
                selected_index = st.selectbox(
                    '학생', student_options, format_func=lambda idx: labels[idx], key='analysis_student_select'
                )
                row = records.loc[selected_index]
                name = str(row.get('성명', ''))
                label = student_label(row)
                with st.expander('원문 보기', expanded=False):
                    for c in ['창체', '교과세특', '행발']:
                        st.markdown(f'**{c}**')
                        st.write(row.get(c, '') or '자료 없음')
                card_close()

            with r:
                card_open(f'{label} 키워드 분석 · {scope}', '전처리 캐시에서 읽은 TF-IDF와 빈도표를 사용합니다.')
                selected = tfidf_df[student_mask(tfidf_df, row)] if not tfidf_df.empty else pd.DataFrame()
                freq = freq_df[student_mask(freq_df, row)] if not freq_df.empty else pd.DataFrame()
                selected = selected.copy()
                freq = freq.copy()
                if 'TF-IDF' in selected.columns:
                    selected['TF-IDF'] = pd.to_numeric(selected['TF-IDF'], errors='coerce').fillna(0)
                if '빈도' in freq.columns:
                    freq['빈도'] = pd.to_numeric(freq['빈도'], errors='coerce').fillna(0).astype(int)
                wc_dict = dict(zip(selected['단어'], selected['TF-IDF'])) if not selected.empty else dict(zip(freq['단어'], freq['빈도'])) if not freq.empty else {}
                fig = wordcloud_fig(wc_dict)
                if fig:
                    st.pyplot(fig, clear_figure=True, use_container_width=True)
                    st.caption('글자가 클수록 이 학생의 기록에서 상대적으로 두드러지는 TF-IDF 특징어입니다.')
                else:
                    st.info('워드클라우드를 만들 키워드가 없습니다.')
                card_close()

            with st.expander('키워드 상세표', expanded=False):
                tfidf_tab, freq_tab = st.tabs(['TF-IDF 특징어', '단어 빈도'])
                with tfidf_tab:
                    tfidf_columns = [column for column in ['순위', '단어', 'TF-IDF'] if column in selected.columns]
                    st.dataframe(selected[tfidf_columns].head(top_n), use_container_width=True, hide_index=True)
                with freq_tab:
                    freq_columns = [column for column in ['순위', '단어', '빈도'] if column in freq.columns]
                    st.dataframe(freq[freq_columns].head(top_n), use_container_width=True, hide_index=True)

            with st.expander('키워드 근거 원문 추적', expanded=False):
                keyword_candidates = []
                if not selected.empty and '단어' in selected.columns:
                    keyword_candidates += selected['단어'].astype(str).tolist()
                if not freq.empty and '단어' in freq.columns:
                    keyword_candidates += freq['단어'].astype(str).tolist()
                keyword_candidates = list(dict.fromkeys([x for x in keyword_candidates if x]))
                if not keyword_candidates:
                    st.info('근거를 확인할 키워드가 없습니다.')
                else:
                    c1, c2 = st.columns([0.9, 2.1])
                    with c1:
                        selected_keyword = st.selectbox('근거를 확인할 키워드', keyword_candidates)
                        if not evidence_df.empty and '키워드목록' in evidence_df.columns:
                            pattern = rf'(^|,\s*){re.escape(selected_keyword)}($|,\s*)'
                            evd = evidence_df[student_mask(evidence_df, row) & evidence_df['키워드목록'].astype(str).str.contains(pattern, regex=True, na=False)].copy()
                            if not evd.empty:
                                evd['등장횟수'] = evd['키워드목록'].astype(str).apply(lambda x: x.split(', ').count(selected_keyword))
                                evd['강조원문'] = evd['원문']
                        else:
                            evd = keyword_evidence(row, selected_keyword, stop, syn, min_len, effective_analyzer)
                        st.metric('근거 문장 수', len(evd))
                        if not evd.empty:
                            st.dataframe(evidence_summary(evd) if '등장횟수' in evd.columns else evd.groupby('출처').size().reset_index(name='문장수'), use_container_width=True, hide_index=True)
                    with c2:
                        if evd.empty:
                            st.info('해당 키워드를 포함하는 근거 문장을 찾지 못했습니다.')
                        else:
                            st.markdown(evidence_html(evd, selected_keyword, syn), unsafe_allow_html=True)

            # 학과 말뭉치: 사이드바 업로드 → 세션 → 내장 DB 순서로 사용합니다.
            major_df = pd.DataFrame()
            if isinstance(st.session_state.get('major_corpus_df'), pd.DataFrame):
                major_df = st.session_state['major_corpus_df']
            if major_df.empty:
                major_df = load_db()

            if not major_df.empty:
                with st.expander('학과 유사도 / 희망 학과 보완 키워드', expanded=True):
                    text = str(row.get(scope, ''))
                    available_channels = [
                        name for name in ['통합', '학업', '교과', '활동', '적성', '진로']
                        if f'말뭉치_{name}' in major_df.columns
                        and major_df[f'말뭉치_{name}'].astype(str).str.strip().ne('').any()
                    ]
                    if not available_channels:
                        available_channels = ['통합']
                    analysis_channel = st.radio(
                        '학과 분석 관점',
                        available_channels,
                        horizontal=True,
                        help='학과 정보의 어느 영역을 학생부와 비교할지 선택합니다.',
                    )
                    with st.spinner(f'{analysis_channel} 말뭉치와 현재 학생 집단의 유사도 기준을 준비하는 중입니다. 최초 한 번은 시간이 걸릴 수 있습니다.'):
                        thresholds = calibrate_similarity_thresholds(
                            tuple(records[scope].fillna('').astype(str)),
                            tuple(corpus_texts(major_df, analysis_channel)),
                            tuple(sorted(stop)),
                            tuple(sorted(syn.items())),
                            min_len,
                            effective_analyzer,
                        )
                        sim_raw = similarity(
                            text, major_df, stop, syn, min_len, effective_analyzer, 15,
                            channel=analysis_channel,
                        )
                        sim = label_similarity_results(sim_raw, thresholds) if not sim_raw.empty else sim_raw
                    if sim.empty:
                        st.warning('학과 유사도 계산 결과가 없습니다. 학과 말뭉치의 「말뭉치」 또는 학과 설명 열을 확인해 주세요.')
                    else:
                        with st.expander('유사 학과 목록 보기', expanded=False):
                            st.caption(
                                f'현재 학생 {thresholds.get("students", 0):,}명 × 학과 {thresholds.get("majors", 0):,}개의 '
                                f'텍스트 비교 분포를 기준으로 상위 1%를 「적합」, 상위 5%를 「높은 연계」로 표시합니다. '
                                '이는 진로 적합성이나 합격 가능성 판정이 아닙니다.'
                            )
                            st.dataframe(style_similarity_results(sim), use_container_width=True, hide_index=True)

                    name_col = '학과명' if '학과명' in major_df.columns else ('학과' if '학과' in major_df.columns else major_df.columns[0])
                    fil = st.text_input('희망 학과 검색')
                    pool = major_df[major_df[name_col].astype(str).str.contains(fil, na=False)] if fil else major_df
                    sort_columns = [name_col] + (['계열'] if '계열' in pool.columns else [])
                    pool = pool.sort_values(sort_columns, kind='stable').reset_index(drop=True)
                    labels = [f"{x.get('계열','')} | {x.get(name_col,'')}" for _, x in pool.iterrows()]
                    if labels:
                        top_major_label = ''
                        if not sim.empty:
                            top_major = sim.iloc[0]
                            top_major_label = f"{top_major.get('계열', '')} | {top_major.get('학과명', '')}"
                        default_major_index = labels.index(top_major_label) if top_major_label in labels else 0
                        student_key = '|'.join(str(row.get(column, '')) for column in STUDENT_ID_COLUMNS)
                        major_select_key = (
                            f'analysis_target_major::{student_key}::{scope}::{analysis_channel}::{top_major_label}'
                        )
                        lab = st.selectbox(
                            '희망 학과 선택',
                            labels,
                            index=default_major_index,
                            key=major_select_key,
                            help='목록은 가나다순이며, 학생을 처음 선택하면 유사도 1위 학과가 먼저 표시됩니다.',
                        )
                        target = pool.iloc[labels.index(lab)]
                        gap, common, gap_freq, metrics = gap_analysis(
                            text, target, major_df, stop, syn, min_len,
                            effective_analyzer, top_n, channel=analysis_channel,
                        )
                        dash_col, wc_gap_col = st.columns([0.95, 1.25])
                        with dash_col:
                            render_similarity_dashboard(
                                metrics,
                                thresholds,
                                len(gap) if not gap.empty else 0,
                            )
                        with wc_gap_col:
                            st.markdown('**보완 키워드 워드클라우드**')
                            fig2 = wordcloud_fig(gap_freq)
                            if fig2:
                                st.pyplot(fig2, clear_figure=True)
                            else:
                                st.info('보완 키워드가 없습니다.')
                        gt, ct = st.tabs(['보완 키워드', '공통 키워드'])
                        with gt:
                            st.dataframe(gap, use_container_width=True, hide_index=True)
                        with ct:
                            st.dataframe(common, use_container_width=True, hide_index=True)
                    else:
                        st.warning('조건에 맞는 학과가 없습니다.')
            else:
                st.info('왼쪽 사이드바에서 학과 말뭉치 DB/CSV/XLSX를 넣거나, 3번 말뭉치 관리 탭에서 내장 DB를 저장하면 학과 유사도 분석이 가능합니다.')
            render_footer()

    # ------------------------------------------------------------------
    # 2. 학생부 전처리: 원본 창체/세특/행발 파일은 여기에서만 업로드합니다.
    # ------------------------------------------------------------------
    with tab_pre:
        st.subheader('학생부 전처리')
        st.caption('원본 학생부를 한 번 처리해 학생별 특징어·빈도·근거문장 캐시를 만듭니다.')
        render_help('전처리·형태소 안내', """
전처리는 원본 창체·교과세특·행발 파일을 학생별로 병합하고, 형태소 분석·빈도·TF-IDF·근거문장 인덱스를 미리 계산하는 과정입니다. **분석은 한 번, 조회는 여러 번** 하기 위한 단계입니다.

- **Kiwi**: 한국어 문장 분석 품질과 설치 편의의 균형이 좋아 기본값으로 사용합니다.
- **MeCab**: 빠른 한국어 형태소 분석기지만 별도 한국어 패키지·사전과 Windows 실행 구성요소가 필요합니다. 설치된 환경에서만 선택할 수 있습니다.
- **간이 토큰화**: 별도 분석기 없이 빠르게 처리하지만 조사·어미가 붙은 표현이나 잡음이 더 남을 수 있습니다.
- 전처리 결과에는 사용한 방식이 기록됩니다. 이미 만든 캐시는 나중에 선택값을 바꿔도 자동으로 다른 방식이 되지 않습니다.
- 방식을 바꾸려면 원본 파일을 다시 넣고 전처리 결과를 새로 만들어야 합니다.

전처리 결과에는 학생 식별정보와 원문이 포함될 수 있으므로 원본 학생부와 같은 수준으로 보호하세요.
""")
        up_col, run_col = st.columns([1, 1])
        with up_col:
            st.markdown('#### 원본 학생부 파일 업로드')
            render_student_file_help()
            f_ch = st.file_uploader('창체 파일(.xlsx, 복수 선택 가능)', type='xlsx', accept_multiple_files=True, key='pre_changche')
            f_se = st.file_uploader('교과세특 파일(.xlsx, 복수 선택 가능)', type='xlsx', accept_multiple_files=True, key='pre_setuk')
            f_ha = st.file_uploader('행발 파일(.xlsx, 복수 선택 가능)', type='xlsx', accept_multiple_files=True, key='pre_haengbal')
            st.caption('학년·반은 엑셀 상단 또는 파일명에서 자동 인식합니다. 파일명 예: `3학년 1반 창체.xlsx` 또는 `3-1반 창체.xlsx`')
            merged = pd.DataFrame()
            msgs = []
            if any([f_ch, f_se, f_ha]):
                upload_signature = uploaded_records_signature(f_ch, f_se, f_ha)
                if upload_signature != st.session_state.get('uploaded_records_signature'):
                    with st.status('학생부 파일을 읽고 있습니다.', expanded=True) as upload_status:
                        upload_progress = st.progress(0, text='업로드 파일을 확인하는 중입니다.')

                        def update_upload_progress(fraction: float, message: str) -> None:
                            upload_progress.progress(min(max(fraction, 0.0), 1.0), text=message)

                        merged, msgs = parse_uploaded_records(
                            f_ch, f_se, f_ha, progress_callback=update_upload_progress
                        )
                        upload_status.update(
                            label=f'파일 읽기 완료 · 학생 {len(merged)}명',
                            state='complete',
                            expanded=False,
                        )
                    st.session_state['uploaded_records'] = merged
                    st.session_state['upload_messages'] = msgs
                    st.session_state['uploaded_records_signature'] = upload_signature
                    st.session_state.pop('student_cache', None)
                    st.session_state.pop('student_cache_name', None)
                else:
                    merged = st.session_state.get('uploaded_records', pd.DataFrame())
                    msgs = st.session_state.get('upload_messages', [])
            elif isinstance(st.session_state.get('uploaded_records'), pd.DataFrame):
                merged = st.session_state['uploaded_records']
                msgs = st.session_state.get('upload_messages', [])

            if merged.empty:
                st.info('창체·교과세특·행발 파일을 넣으면 병합 현황이 표시됩니다.')
            else:
                st.success(' / '.join(msgs) + f' / 병합 {len(merged)}명')
                with st.expander('병합 자료 미리보기', expanded=False):
                    st.dataframe(merged[['학년', '반', '번호', '성명', '창체', '교과세특', '행발']].head(50), use_container_width=True, hide_index=True)

        with run_col:
            st.markdown('#### 전처리 실행 및 저장')
            selected_analyzer = st.selectbox(
                '형태소 분석기',
                ['Kiwi', 'MeCab', '간이 토큰화'],
                key='preprocess_analyzer',
                help='전처리 결과에 기록되며, 이후 학생 분석에서도 같은 방식을 사용합니다.',
            )
            with st.expander('형태소 분석기 설치 안내', expanded=False):
                st.markdown(
                    '- 릴리즈 기본 포함: `Kiwi` + `kiwipiepy_model` (권장)\n'
                    '- 릴리즈 기본 포함: `간이 토큰화` (추가 설치 없음)\n'
                    '- 기본 미포함(선택): `MeCab`\n'
                    '\n'
                    'MeCab을 사용하려면:\n'
                    '1. Python 래퍼(`mecab_ko` 또는 `mecab`)를 설치하세요.\n'
                    '2. Windows용 MeCab 엔진/사전을 설치하세요.\n'
                    '3. 앱을 재시작한 뒤 분석기 목록에서 `MeCab`을 선택하세요.\n'
                    '\n'
                    '설치가 어렵거나 환경 제약이 있으면 기본 경로인 `Kiwi`를 사용하세요.'
                )
            selected_analyzer_available = analyzer_available(selected_analyzer)
            unavailable_reason = analyzer_unavailable_reason(selected_analyzer)
            if selected_analyzer == 'Kiwi' and not selected_analyzer_available:
                st.warning(f'Kiwi 분석기를 사용할 수 없습니다. {unavailable_reason or "kiwipiepy_model이 누락되었거나 사용할 수 없습니다."}')
            elif selected_analyzer == 'Kiwi':
                st.caption('권장 기본값 · 한국어 명사 추출이 비교적 정교합니다.')
            elif selected_analyzer == 'MeCab' and not selected_analyzer_available:
                st.warning('현재 실행 환경에는 한국어 MeCab이 설치되어 있지 않아 선택할 수 없습니다. Kiwi 또는 간이 토큰화를 사용해 주세요.')
            elif selected_analyzer == 'MeCab':
                st.caption('설치 확인됨 · 빠른 한국어 형태소 분석을 사용합니다.')
            else:
                st.caption('가장 빠름 · 별도 형태소 분석기 없이 간단한 규칙으로 단어를 나눕니다.')
            fmt = st.radio('저장 형식', ['CSV 묶음(zip)', 'Excel(xlsx)', 'SQLite(db)'], horizontal=True)
            merged = st.session_state.get('uploaded_records', pd.DataFrame())
            if not isinstance(merged, pd.DataFrame) or merged.empty:
                st.warning('먼저 왼쪽에 원본 학생부 파일을 업로드해야 전처리를 실행할 수 있습니다.')
            else:
                st.write(f'대상 학생 수: **{len(merged)}명** / 분석 범위: **{scope}**')
                if st.button('전처리 실행', type='primary', disabled=not selected_analyzer_available):
                    with st.status('학생부 전처리를 시작합니다.', expanded=True) as preprocess_status:
                        preprocess_progress = st.progress(0, text='전처리 준비 중입니다.')

                        def update_preprocess_progress(fraction: float, message: str) -> None:
                            preprocess_progress.progress(min(max(fraction, 0.0), 1.0), text=message)

                        cache = build_student_cache(
                            merged,
                            scope,
                            stop,
                            syn,
                            min_len,
                            selected_analyzer,
                            top_n,
                            progress_callback=update_preprocess_progress,
                        )
                        st.session_state['student_cache'] = cache
                        st.session_state['student_cache_name'] = '현재 세션에서 생성한 캐시'
                        save_student_cache_db(cache)
                        preprocess_status.update(
                            label=f'전처리 완료 · 학생 {len(merged)}명',
                            state='complete',
                            expanded=False,
                        )
                    st.success('전처리가 완료되었습니다. 아래에서 저장 파일을 내려받을 수 있습니다.')

                cache = st.session_state.get('student_cache')
                if isinstance(cache, dict) and 'records' in cache:
                    m1, m2, m3 = st.columns(3)
                    m1.metric('캐시 학생 수', len(cache.get('records', pd.DataFrame())))
                    m2.metric('TF-IDF 행 수', len(cache.get('tfidf', pd.DataFrame())))
                    m3.metric('근거문장 행 수', len(cache.get('evidence', pd.DataFrame())))
                    if fmt == 'CSV 묶음(zip)':
                        st.download_button('전처리 결과 ZIP 다운로드', cache_to_zip(cache), 'student_record_cache.zip', 'application/zip')
                    elif fmt == 'Excel(xlsx)':
                        st.download_button('전처리 결과 Excel 다운로드', cache_to_excel(cache), 'student_record_cache.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    else:
                        st.download_button('전처리 결과 DB 다운로드', cache_to_db(cache), 'student_record_cache.db', 'application/octet-stream')
        render_footer()

    # ------------------------------------------------------------------
    # 3. 말뭉치 관리
    # ------------------------------------------------------------------
    with tab_corpus:
        st.subheader('말뭉치 관리')
        st.caption('커리어넷 학과 상세정보를 수집·가공하거나 기존 말뭉치를 확인하고 내장 DB로 저장합니다.')
        render_help('말뭉치·API 안내', """
**말뭉치**는 학생부와 비교할 기준 텍스트입니다. 이 앱은 커리어넷 공개 학과정보를 학업·교과·활동·적성·진로 영역으로 나누어 사용합니다.

- 커리어넷 갱신 시 학생부 원문이나 학생 분석 결과를 전송하지 않습니다.
- 외부로 전송되는 값은 API 키와 공개 학과정보를 요청하는 데 필요한 항목입니다.
- API 키는 화면에서 입력하면 현재 실행 중에만 사용되며 앱 코드나 DB에 저장하지 않습니다. 키를 GitHub·문서·공개 화면에 남기지 마세요.
- `오류·누락 학과만 보완`은 정상 자료를 재사용하므로 빠르고, `전체 학과 새로 갱신`은 최신 자료를 모두 다시 받습니다.
- 수집 직후 결과는 현재 세션에만 있습니다. 검토 후 `현재 말뭉치를 내장 DB로 저장`해야 다음 실행에서도 사용할 수 있습니다.
- 학과정보의 표현과 갱신 시점이 바뀌면 유사도 결과도 달라질 수 있습니다.
""")
        save_format = st.selectbox('말뭉치 저장 형식', ['DB', 'CSV', 'Excel'], key='corpus_save_format')
        major_df = pd.DataFrame()
        if isinstance(st.session_state.get('major_corpus_df'), pd.DataFrame):
            major_df = st.session_state['major_corpus_df']
            st.success(f'사이드바에서 불러온 말뭉치: {len(major_df)}개')
        elif DB_PATH.exists():
            major_df = load_db()
            st.success(f'내장 DB 사용 가능: {DB_PATH.name} / {len(major_df)}개')
        else:
            st.warning('내장 DB가 없습니다. 왼쪽 사이드바에서 CSV/XLSX/DB를 업로드해 저장하세요.')

        with st.expander('커리어넷에서 학과 말뭉치 수집·갱신', expanded=major_df.empty):
            st.write(
                'API 키는 이번 실행 중에만 사용하며 파일에 저장하지 않습니다. '
                '수집한 결과는 확인 후 내장 DB로 저장할 수 있습니다.'
            )
            api_key = st.text_input(
                '커리어넷 API 키',
                value=os.getenv('CAREERNET_API_KEY', ''),
                type='password',
                key='careernet_api_key',
            )
            refresh_mode = st.radio(
                '갱신 범위',
                ['오류·누락 학과만 보완', '전체 학과 새로 갱신'],
                horizontal=True,
                help='오류·누락 보완은 이미 정상 수집된 학과를 재사용해 훨씬 빠릅니다.',
            )
            if st.button('커리어넷 학과 말뭉치 수집 시작', type='primary'):
                if not api_key.strip():
                    st.error('커리어넷 API 키를 입력해 주세요.')
                else:
                    try:
                        with st.status('커리어넷 학과정보를 수집하고 있습니다.', expanded=True) as corpus_status:
                            corpus_progress = st.progress(0, text='학과 목록 요청 준비 중입니다.')

                            def update_corpus_progress(fraction: float, message: str) -> None:
                                corpus_progress.progress(min(max(fraction, 0.0), 1.0), text=message)

                            collected = collect_careernet_corpus(
                                api_key,
                                existing=major_df,
                                refresh_all=refresh_mode == '전체 학과 새로 갱신',
                                progress_callback=update_corpus_progress,
                            )
                            quality = corpus_quality(collected)
                            corpus_status.update(
                                label=f'수집 완료 · 정상 {quality["complete"]}개 · 오류 {quality["errors"]}개',
                                state='complete' if quality['errors'] == 0 else 'error',
                                expanded=quality['errors'] > 0,
                            )
                        st.session_state['major_corpus_df'] = collected
                        st.session_state['major_corpus_name'] = '커리어넷에서 현재 세션에 수집한 말뭉치'
                        save_db(collected)
                        major_df = collected
                        prepare_major_index.clear()
                        st.success('수집과 영역별 말뭉치 가공이 완료되었습니다. 아래에서 내용을 확인한 뒤 내장 DB로 저장하세요.')
                    except Exception as exc:
                        st.error(f'커리어넷 말뭉치 수집에 실패했습니다: {exc}')

        if not major_df.empty:
            major_df = enrich_existing_corpus(major_df)
            quality = corpus_quality(major_df)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric('전체 학과', f'{quality["total"]:,}개')
            q2.metric('정상 수집', f'{quality["complete"]:,}개')
            q3.metric('수집 오류', f'{quality["errors"]:,}개')
            q4.metric('마지막 갱신', quality['updated_at'] or '기록 없음')
            with st.expander('필드별 데이터 품질 확인', expanded=False):
                quality_rows = [
                    {'필드': field, '자료 있음': count, '누락': quality['total'] - count}
                    for field, count in quality.get('field_counts', {}).items()
                ]
                st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
            st.dataframe(major_df.head(30), use_container_width=True)
            if st.button('현재 말뭉치를 내장 DB로 저장'):
                save_db(major_df)
                st.success('data/major_corpus.db로 저장했습니다. 이후 오프라인 분석에 사용됩니다.')
            if save_format == 'DB':
                save_db(major_df, DATA_DIR / '_download.db')
                st.download_button('DB 다운로드', (DATA_DIR / '_download.db').read_bytes(), 'major_corpus.db')
            else:
                b, m, n = df_bytes(major_df, save_format)
                st.download_button(f'{save_format} 다운로드', b, n, m)
        render_footer()

    # ------------------------------------------------------------------
    # 4. 분석 사전
    # ------------------------------------------------------------------
    with tab_dict:
        st.subheader('분석 사전')
        st.caption('분석에서 제외할 단어와 같은 의미로 통일할 표현을 관리합니다.')
        render_help('분석 사전 안내', """
- **불용어**: 분석에서 제외할 단어입니다. 지나치게 많이 넣으면 의미 있는 단어까지 사라질 수 있습니다.
- **표현 통일**: `AI=인공지능`처럼 서로 다른 표기를 하나로 합칩니다. 한 줄에 하나씩 `원래표현=통일표현` 형식으로 적습니다.
- 사전을 바꾸면 이후 전처리와 학과 분석 결과가 달라질 수 있습니다. 결과를 비교할 때는 사용한 사전 버전을 함께 관리하세요.
- 학생의 표현을 평가하거나 특정 결과를 만들기 위해 임의로 단어를 과도하게 제외하지 마세요.
""")
        dictionary_candidates = dictionary_stopword_candidates(
            st.session_state.get('student_cache'), read_stopwords()
        )
        st.markdown('#### 현재 학생부 기반 사전 점검')
        if dictionary_candidates.empty:
            st.info('학생부 전처리 캐시를 불러오면 여러 학생에게 반복되는 불용어 검토 후보를 여기에서 확인할 수 있습니다.')
        else:
            st.caption('전체 학생의 35% 이상에게 반복된 단어입니다. 교육적으로 의미 있는 표현일 수 있으므로 원문을 확인한 뒤 필요한 단어만 불용어에 추가하세요.')
            st.dataframe(dictionary_candidates.head(30), use_container_width=True, hide_index=True)
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            txt = st.text_area('stopwords.txt', STOPWORDS_PATH.read_text(encoding='utf-8'), height=360)
            if st.button('불용어 저장'):
                STOPWORDS_PATH.write_text(txt, encoding='utf-8')
                st.success('저장했습니다.')
        with c2:
            txt2 = st.text_area('synonyms.txt', SYNONYMS_PATH.read_text(encoding='utf-8'), height=360)
            if st.button('표현 통일 저장'):
                SYNONYMS_PATH.write_text(txt2, encoding='utf-8')
                st.success('저장했습니다.')
        render_footer()

    # ------------------------------------------------------------------
    # 5. 사용 안내
    # ------------------------------------------------------------------
    with tab_help:
        render_full_user_guide()
        render_footer()


if __name__ == '__main__':
    main()
