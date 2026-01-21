#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For many packages, infer a "compatible version span" (from min_version to max_version) for a target repo,
using:
    1) extracted_calls.jsonl
    2) per-package lifecycle DB folders (events + optional index), i.e.:
        <db_root>/<package>/_lifecycle_events.jsonl  (or _lifespan_events.jsonl)
        <db_root>/<package>/_index.json              (optional 1)
        <db_root>/<package>/<version>/...            (optional 2: used for version discovery)

- Requirements: openai, pydantic, packaging

- How to run:
    (env var로 OPENAI API KEY 설정 이후)
    $ python span_inference_with_llm.py \
        --calls extracted_calls.jsonl \
        --db-root /path/to/out \
        --out llm_spans_output.jsonl \
        --model gpt-5-nano \
        --packages torch (optional, 패키지 따로따로 돌리고싶으면 원하는 패키지 이름만 붙일 것)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from packaging.version import Version, InvalidVersion
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class SpanResult(BaseModel):
    """llm 최종 출력 schema"""
    package: str
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: str = Field(..., description="llm|insufficient")
    caveats: List[str] = Field(default_factory=list)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """jsonl파일을 한줄씩 dict로 읽어옴"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """records to jsonl"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def vparse(v: str) -> Optional[Version]:
    """문자열 버전을 packaging.version.Version으로 parse"""
    try:
        return Version(v)
    except InvalidVersion:
        return None

def safe_sort_versions(vs: List[str]) -> List[str]:
    """버전 문자열 list 안전하게 정렬"""
    def key(x: str):
        vx = vparse(x)
        return (vx is None, vx or Version("0"), x)
    return sorted(vs, key=key)

def prev_version(universe: List[str], v: str) -> Optional[str]:
    """universe에서 v의 바로 이전 버전 return"""
    if v not in universe:
        return None
    i = universe.index(v)
    return universe[i - 1] if i > 0 else None


@dataclasses.dataclass(frozen=True)
class NormCall:
    """extracted_calls.jsonl을 패키지 중심으로 minimize해서 llm한테 전달될 토큰수 간소화"""
    pkg_root: str
    qualname_like: str
    kw_names: Tuple[str, ...]


def normalize_call(rec: Dict[str, Any]) -> NormCall:
    """
    extracted_calls.jsonl 1레코드를 표준화:
        - pkg_root: module_root_guess 우선, 없으면 callee_chain[0]
        - qualname_like: (root.) + join(callee_chain)
        - kw_names: 키워드 인자 이름들
    """
    chain = rec.get("callee_chain") or []
    if not isinstance(chain, list):
        chain = [str(chain)]
    chain = [str(x) for x in chain if x is not None]

    root = (rec.get("module_root_guess") or "").strip()
    pkg_root = root if root else (chain[0] if chain else "")
    base = ".".join(chain)
    qual = f"{root}.{base}" if root else base

    kw = rec.get("kw_names") or []
    if not isinstance(kw, list):
        kw = [str(kw)]
    kw = tuple(str(x) for x in kw if x is not None and str(x) != "")

    return NormCall(pkg_root=pkg_root, qualname_like=qual, kw_names=kw)


def load_calls_all(calls_path: Path) -> Dict[str, List[NormCall]]:
    """calls jsonl 전체를 읽고 pkg_root별로 그룹핑해서 return"""
    groups: Dict[str, List[NormCall]] = defaultdict(list)
    for rec in iter_jsonl(calls_path):
        c = normalize_call(rec)
        if c.pkg_root:
            groups[c.pkg_root].append(c)
    return groups


def find_package_dirs(db_root: Path) -> Dict[str, Path]:
    """db_root 경로에서 lifecycle events 파일 있는 pkg dir을 찾음(db_root 자체가 패키지 폴더인 경우 포함)"""
    db_root = db_root.resolve()
    if (db_root / "_lifecycle_events.jsonl").exists() or (db_root / "_lifespan_events.jsonl").exists():
        return {db_root.name: db_root}

    out: Dict[str, Path] = {}
    for child in db_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "_lifecycle_events.jsonl").exists() or (child / "_lifespan_events.jsonl").exists():
            out[child.name] = child
    return out


def pick_events_file(pkg_dir: Path) -> Optional[Path]:
    """pkg dir에서 event jsonl 파일 경로 고름(_lifecycle_events 우선)"""
    p1 = pkg_dir / "_lifecycle_events.jsonl"
    p2 = pkg_dir / "_lifespan_events.jsonl"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def discover_universe(pkg_dir: Path, versions_from_events: List[str]) -> List[str]:
    """
    version universe 구성, 우선순위는 아래와 같이 이뤄짐
    1) _index.json의 versions
    2) 버전명 디렉토리
    3) events log에 등장한 version 집합
    """
    vs = []
    for child in pkg_dir.iterdir():
        if child.is_dir() and vparse(child.name) is not None:
            vs.append(child.name)
    if vs:
        return safe_sort_versions(vs)
    
    idx = pkg_dir / "_index.json"
    if idx.exists():
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
            raw = data.get("versions")
            if isinstance(data, dict) and isinstance(data.get("versions"), list):
                cleaned = []
                for item in raw:
                    if isinstance(item, str):
                        cleaned.append(item)
                    elif isinstance(item, dict) and "version" in item:
                        cleaned.append(str(item["version"]))
                cleaned = [v for v in cleaned if vparse(v) is not None]
                if cleaned:
                    return safe_sort_versions(cleaned)
        except Exception:
            pass
    
    cleaned = [v for v in versions_from_events if vparse(str(v)) is not None]
    return safe_sort_versions(list(dict.fromkeys(cleaned)))



@dataclasses.dataclass
class Event:
    """lifecycle event record"""
    id: str
    version: str
    change: str
    qualname: Optional[str]
    details: List[Dict[str, Any]]


def load_events(pkg_dir: Path) -> Tuple[List[Event], List[str]]:
    events_file = pick_events_file(pkg_dir)  # Optional[Path]로 바뀐 상태
    if events_file is None:
        return [], []

    raw = list(iter_jsonl(events_file))

    id_to_qual: Dict[str, str] = {}
    versions: List[str] = []
    for r in raw:
        if r.get("version") is not None:
            versions.append(str(r["version"]))
        i = str(r.get("id", ""))
        q = r.get("qualname")
        if i and isinstance(q, str) and q:
            id_to_qual[i] = q

    events: List[Event] = []
    for r in raw:
        i = str(r.get("id", ""))
        v = str(r.get("version", ""))
        c = str(r.get("change", ""))
        q = r.get("qualname")
        if not (isinstance(q, str) and q):
            q = id_to_qual.get(i)
        d = r.get("details") or []
        events.append(Event(id=i, version=v, change=c, qualname=q, details=d))

    return events, versions


def param_delta(e: Event) -> Dict[str, Any]:
    """modified event에서 added/removed/renames 등 param modification만 추출하고, 없으면 empty 유지"""
    if e.change != "modified":
        return {}
    for d in e.details or []:
        if d.get("field") == "params":
            p = d.get("params") or {}
            return p if isinstance(p, dict) else {}
    return {}


def split_qual(q: str) -> List[str]:
    """qualname split"""
    return [p for p in q.split(".") if p]

def build_suffix_index(qualnames: Iterable[str], max_suffix_parts: int) -> Dict[str, Set[str]]:
    """
    full qualname들로 suffix index 만듬
    예) a.b.C.f는 suffix: f / C.f / b.C.f ...
    """
    idx: Dict[str, Set[str]] = defaultdict(set)
    for q in qualnames:
        parts = split_qual(q)
        for k in range(1, min(max_suffix_parts, len(parts)) + 1):
            idx[".".join(parts[-k:])].add(q)
    return idx

def candidates(call_qual: str, idx: Dict[str, Set[str]], max_candidates: int = 10) -> List[str]:
    """call qualname_like에 대해 suffix를 1-5토큰 길이로 매칭하여 후보 full qualname들을 반환"""
    parts = split_qual(call_qual)
    out: List[str] = []
    seen: Set[str] = set()
    for k in range(min(5, len(parts)), 0, -1):
        suf = ".".join(parts[-k:])
        for q in sorted(idx.get(suf, set())):
            if q not in seen:
                out.append(q)
                seen.add(q)
        if len(out) >= max_candidates:
            break
    return out[:max_candidates]


def build_kw_availability(universe: List[str], events: List[Event]) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    """
    event에서 symbol별 kw의 존재구간 추정
    - added at v      -> earliest=v
    - removed at v    -> last=prev(v)
    - renames old->new=> old.last=prev(v), new.earliest=v
    
    반환값은:
    avail[symbol][kw] = {"earliest": str|None, "last": str|None}
    """
    avail: Dict[str, Dict[str, Dict[str, Optional[str]]]] = defaultdict(lambda: defaultdict(lambda: {"earliest": None, "last": None}))
    pos = {v: i for i, v in enumerate(universe)}

    def ekey(e: Event) -> Tuple[int, str]:
        return (pos.get(e.version, 10**9), e.version)

    for e in sorted(events, key=ekey):
        if not e.qualname:
            continue
        sym = e.qualname
        
        # symbol level lifecycle record
        if e.change == "added":
            if avail[sym]["@symbol"]["earliest"] is None:
                avail[sym]["@symbol"]["earliest"] = e.version
        elif e.change == "removed":
            pv = prev_version(universe, e.version)
            avail[sym]["@symbol"]["last"] = pv
        
        p = param_delta(e)
        if not p:
            continue
        sym = e.qualname

        added = p.get("added") or []
        removed = p.get("removed") or []
        renames = p.get("renames") or {}

        if isinstance(added, list):
            for kw in added:
                kw = str(kw)
                if avail[sym][kw]["earliest"] is None:
                    avail[sym][kw]["earliest"] = e.version

        if isinstance(removed, list):
            pv = prev_version(universe, e.version)
            for kw in removed:
                avail[sym][str(kw)]["last"] = pv

        if isinstance(renames, dict):
            pv = prev_version(universe, e.version)
            for old, new in renames.items():
                avail[sym][str(old)]["last"] = pv
                if avail[sym][str(new)]["earliest"] is None:
                    avail[sym][str(new)]["earliest"] = e.version

    return avail


def summarize_calls(pkg_calls: List[NormCall], topk: int = 40) -> Dict[str, Any]:
    """llm 입력토큰개수 줄이기 위해 call qualname/kw 빈도 상위 topk를 요약"""
    q = Counter()
    kw = Counter()
    for c in pkg_calls:
        q[c.qualname_like] += 1
        for k in c.kw_names:
            kw[k] += 1
    return {
        "total_calls": len(pkg_calls),
        "calls_with_kw": sum(1 for c in pkg_calls if c.kw_names),
        "top_qualnames": q.most_common(topk),
        "top_keywords": kw.most_common(topk),
    }


def build_constraints(
    universe: List[str],
    pkg_calls: List[NormCall],
    events: List[Event],
    max_suffix_parts: int,
    max_constraints: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    관측된 kw use를 기반으로 DB의 params delta와 결합해서 constraints 만듬
    
    - call -> 후보 symbol(suffix matching) -> kw별 earliest/last lookup
    
    반환값은:
        - constraints: [{"symbol","kw","earliest","last"}, ...]
        - stats: 매칭 hit/miss, constraints 개수 등
    """
    qualnames = [e.qualname for e in events if e.qualname]
    suf_idx = build_suffix_index(qualnames, max_suffix_parts=max_suffix_parts)
    avail = build_kw_availability(universe, events)
    
    informative_pool = []
    uninformative_pool = []
    
    hits = 0
    misses = 0
    import_hits = 0
    calls_without_kw = 0
    kw_total = 0
    kw_seen: Set[str] = set()
    cand_sizes: List[int] = list()
    candidate_truncated_calls = 0
    miss_qual = Counter()
    avail_lookups = 0
    avail_missing = 0
    
    K = 3
    
    def bucket(n: int) -> str:
        if n == 0: return "0"
        if n == 1: return "1"
        if n == 2: return "2"
        if n == 3: return "3"
        if n <= 5: return "4-5"
        if n <= 10: return "6-10"
        return "11+"

    for c in pkg_calls:
        if c.kw_names:
            kw_total += len(c.kw_names)
            for k in c.kw_names:
                kw_seen.add(k)
        else:
            calls_without_kw += 1
        
        cands = candidates(c.qualname_like, suf_idx)
        cand_sizes.append(len(cands))
        
        if cands:
            hits += 1
            if len(cands) > K:
                candidate_truncated_calls += 1
        
            for sym in cands[:K]:
                sym_av = avail.get(sym, {})
                sym_lifecycle = sym_av.get("@symbol")
                if sym_lifecycle:
                    avail_lookups += 1
                    e_sym, l_sym = sym_lifecycle.get("earliest"), sym_lifecycle.get("last")
                    item = {"symbol": sym, "kw": None, "earliest": e_sym, "last": l_sym, "note": "symbol_existence"}
                    if e_sym is not None or l_sym is not None:
                        informative_pool.append(item)
                    else:
                        uninformative_pool.append(item)
                    
                for kw in c.kw_names:
                    avail_lookups += 1
                    a = sym_av.get(kw)
                    e_kw, l_kw = (a.get("earliest"), a.get("last")) if a else (None, None)
                    if a is None:   avail_missing += 1
                
                    item = {"symbol": sym, "kw": kw, "earliest": e_kw, "last": l_kw}
                    if e_kw is not None or l_kw is not None:
                        informative_pool.append(item)
                    else:
                        uninformative_pool.append(item)
    
        else:
            misses += 1
            found_import_mod = False
            parts = split_qual(c.qualname_like)
            
            for i in range(len(parts) - 1, 0, -1):
                mod_path = ".".join(parts[:i])
                if mod_path in avail:
                    mod_lifecycle = avail[mod_path].get("@symbol")
                    if mod_lifecycle:
                        avail_lookups += 1
                        e_mod, l_mod = mod_lifecycle.get("earliest"), mod_lifecycle.get("last")
                        item = {"symbol": mod_path, "kw": None, "earliest": e_mod, "last": l_mod, "note": "import_existence"}
                        
                        if e_mod is not None or l_mod is not None:
                            informative_pool.append(item)
                            import_hits += 1
                            found_import_mod = True
                        else:
                            uninformative_pool.append(item)
                        
                        if found_import_mod:    break
            
            if not found_import_mod:
                miss_qual[c.qualname_like] += 1
    
    final_constraints = []
    
    if len(informative_pool) <= max_constraints:
        final_constraints.extend(informative_pool)
    else:
        step = len(informative_pool) / max_constraints
        for i in range(max_constraints):
            idx = int(i * step)
            if idx < len(informative_pool):
                final_constraints.append(informative_pool[idx])
    
    remaining_slots = max_constraints - len(final_constraints)
    if remaining_slots > 0 and uninformative_pool:
        step = len(uninformative_pool) / remaining_slots
        for i in range(remaining_slots):
            idx = int(i * step)
            if idx < len(uninformative_pool):
                final_constraints.append(uninformative_pool[idx])
    
    hist = defaultdict(int)
    for n in cand_sizes:    hist[bucket(n)] += 1

    stats = {
        # basic stats
        "calls_total": len(pkg_calls),
        "calls_with_kw": len(pkg_calls) - calls_without_kw,
        "mapping_hits": hits,
        "mapping_misses": misses,
        "import_existence_hits": import_hits,
        "constraints_emitted": len(final_constraints),
        "universe_size": len(universe),
        
        # kw/call property stats
        "kw_total": kw_total,
        "kw_unique": len(kw_seen),

        # candidate matching quality stats
        "candidate_bucket_hist": dict(hist),
        "max_candidates": max(cand_sizes) if cand_sizes else 0,
        "avg_candidates": (statistics.mean(cand_sizes) if cand_sizes else 0.0),
        "candidate_truncated_calls": candidate_truncated_calls,
        "top_missing_qualnames": miss_qual.most_common(10),

        # constraint info stats
        "avail_lookups": avail_lookups,
        "avail_missing": avail_missing,
        "informative_constraints": len(informative_pool),
        "uninformative_constraints": len(uninformative_pool),
    }
    return final_constraints, stats


def select_relevant_events(events: List[Event], constraints: List[Dict[str, Any]], max_events: int) -> List[Dict[str, Any]]:
    """
    llm에 input으로 전달할 event를 constraints 관련 symbol 중심으로 축약
    - 우선 modified+params(delta 존재) event 추가,
    - 이것도 없으면 target symbol의 임의 이벤트 약간 추가
    """
    target_syms = {c["symbol"] for c in constraints}
    picked: List[Dict[str, Any]] = []
    for e in events:
        if e.qualname and e.qualname in target_syms and e.change == "modified":
            if param_delta(e):
                picked.append({"id": e.id, "version": e.version, "change": e.change, "qualname": e.qualname, "details": e.details})
        if len(picked) >= max_events:
            break
    if not picked:
        for e in events:
            if e.qualname and e.qualname in target_syms:
                picked.append({"id": e.id, "version": e.version, "change": e.change, "qualname": e.qualname, "details": e.details})
            if len(picked) >= max_events:
                break
    return picked


def call_llm_span(
    client: OpenAI,
    model: str,
    evidence_pack: Dict[str, Any],
    cache_dir: Path,
    max_retries: int,
) -> SpanResult:
    """
    llm에게 input으로 evidence_pack을 전달하여 SpanResult를 output으로 받음
    동일 evidence_pack이면 SHA256 해시 기반 캐시를 사용해 다시 돌아가는 것 방지
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = hashlib.sha256(json.dumps(evidence_pack, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{evidence_pack['package']}-{fp}.json"

    if cache_path.exists():
        return SpanResult.model_validate_json(cache_path.read_text(encoding="utf-8"))

    system = (
        """Rules:
            - You MUST choose min_version and max_version from version_universe (or return null ONLY if version_universe is empty).
            - Prefer returning a span even if it is wide.
            - If evidence is weak, missing, or contradictory, do NOT return null bounds.
                Instead, return the widest possible span within version_universe (typically min=first, max=last),
                set confidence=0.0, set method="fallback_full_range", and explain why in caveats.
            - If you can confidently narrow the span, return a narrower min/max and set confidence accordingly.
            - Output must be valid JSON matching the provided schema.
            - If you choose 'fallback_full_range', you MUST specify which symbols or versions caused the conflict in the 'caveats' field.
        """
    )

    for attempt in range(max_retries):
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(evidence_pack, ensure_ascii=False)},
                ],
                text_format=SpanResult,
            )
            parsed: SpanResult = resp.output_parsed
            cache_path.write_text(parsed.model_dump_json(ensure_ascii=False), encoding="utf-8")
            return parsed
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** attempt, 20))


def clamp_to_universe(universe: List[str], v: Optional[str], kind: str) -> Tuple[Optional[str], Optional[str]]:
    """
    llm output으로 universe 밖의 version이 나오면 universe 내부에서 가장 가까운 값으로 자동변환시킴
    - kind == "min_version": v 이상인 첫 버전으로 올림
    - kind == "max_version": v 이하인 마지막 버전으로 내림
    
    반환값은:
    (clamped_value, note_message_or_None)
    """
    if not v or v in universe:
        return v, None
    vv = vparse(v)
    if not vv:
        return None, f"{kind} not parseable and not in universe; dropped"
    if kind == "min_version":
        for u in universe:
            uv = vparse(u)
            if uv and uv >= vv:
                return u, f"{kind} clamped to {u}"
        return None, f"{kind} above universe; dropped"
    for u in reversed(universe):
        uv = vparse(u)
        if uv and uv <= vv:
            return u, f"{kind} clamped to {u}"
    return None, f"{kind} below universe; dropped"


def get_dynamic_max_constraints(informative_count: int, default_max: int) -> int:
    """
    informative_constraints 개수에 따라서 max_constraints 제한을 동적으로 조정함
    """
    if informative_count > 1500:
        return 800
    if informative_count > 1000:
        return 500
    if informative_count > 500:
        return 350
    return default_max


def run_one(package: str, pkg_dir: Path, pkg_calls: List[NormCall], args: argparse.Namespace) -> Dict[str, Any]:
    """
    pkg 1개 대상으로 end-to-end:
    1) events/universe load
    2) constraints gen
    3) sufficiency(min_calls/min_constraints/universe size) check -> 부족하면 insufficient
    4) evidence_pack 구성 후 llm call
    5) result clamp/모순 처리
    """
    events, versions_from_events = load_events(pkg_dir)
    universe = discover_universe(pkg_dir, versions_from_events)

    constraints, stats = build_constraints(
        universe=universe,
        pkg_calls=pkg_calls,
        events=events,
        max_suffix_parts=args.max_suffix_parts,
        max_constraints=args.max_constraints,
    )
    
    dynamic_limit = get_dynamic_max_constraints(stats["informative_constraints"], args.max_constraints)
    
    if dynamic_limit > args.max_constraints:
        constraints, stats = build_constraints(
            universe=universe,
            pkg_calls=pkg_calls,
            events=events,
            max_suffix_parts=args.max_suffix_parts,
            max_constraints=dynamic_limit
        )
        stats["dynamic_limit_applied"] = True
        stats["final_max_constraints"] = dynamic_limit
    else:
        stats["dynamic_limit_applied"] = False
        stats["final_max_constraints"] = args.max_constraints
    
    stats["llm_called"] = False
    
    def set_fallback_reason(reason: str) -> None:
        stats.setdefault("fallback_reason", reason)

    caveats: List[str] = []
    
    if len(universe) == 0:
        caveats.append("version_universe is empty")
        set_fallback_reason("empty_universe")
    elif len(universe) == 1:
        caveats.append("version_universe has only 1 version")
        set_fallback_reason("single_version_universe")
    if len(universe) < 2:
        caveats.append("version_universe has <2 versions")
        set_fallback_reason("small_universe")
        
    if stats["calls_total"] < args.min_calls:
        caveats.append(f"too few calls: {stats['calls_total']} < min_calls={args.min_calls}")
        set_fallback_reason("gate_calls_total")
        
    if stats["informative_constraints"] < 1 and stats["constraints_emitted"] < args.min_constraints:
        caveats.append("No informative constraints AND too few total constraints")
        set_fallback_reason("gate_informative_constraints")

    if caveats:
        stats.setdefault("fallback_reason", "unknown")
        if len(universe) >= 2:
            span = SpanResult(
                package=package,
                min_version=universe[0],
                max_version=universe[-1],
                confidence=0.0,
                method="fallback_full_range",
                caveats=caveats + ["fallback applied: evidence insufficient, returning full version universe"],
            )
        elif len(universe) == 1:
            span = SpanResult(
                package=package,
                min_version=universe[0],
                max_version=universe[0],
                confidence=0.0,
                method="fallback_single_version",
                caveats=caveats + ["fallback applied: only one version in universe"],
            )
        else:   #univesre가 아예 아무것도 없을 때
            span = SpanResult(
                package=package,
                min_version=None,
                max_version=None,
                confidence=0.0,
                method="no_db",
                caveats=caveats + ["no fallback possible: empty version_universe"],
        )
        return {"package": package, "span": span.model_dump(), "stats": stats, "version_universe": universe}

    evidence_pack = {
        "package": package,
        "version_universe": universe,
        "call_summary": summarize_calls(pkg_calls, topk=40),
        "constraints": constraints[:160],
        "relevant_events": select_relevant_events(events, constraints, max_events=args.max_events_for_llm),
        "stats": stats,
    }

    if OpenAI is None:
        raise RuntimeError("openai SDK not available. pip install openai")
    
    stats["llm_called"] = True
    client = OpenAI()
    span = call_llm_span(
        client=client,
        model=args.model,
        evidence_pack=evidence_pack,
        cache_dir=Path(args.cache_dir),
        max_retries=args.max_retries,
    )

    notes: List[str] = []
    span.min_version, msg = clamp_to_universe(universe, span.min_version, "min_version")
    if msg:
        notes.append(msg)
    span.max_version, msg = clamp_to_universe(universe, span.max_version, "max_version")
    if msg:
        notes.append(msg)
    if notes:
        span.caveats.extend(notes)

    if span.min_version and span.max_version and span.min_version in universe and span.max_version in universe:
        if universe.index(span.min_version) > universe.index(span.max_version):
            span.caveats.append("inconsistent bounds (min>max); dropped max_version")
            span.max_version = None
            stats.setdefault("fallback_reason", "llm_inconsistency_bounds")

    return {"package": package, "span": span.model_dump(), "stats": stats, "version_universe": universe}


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True, type=str)
    ap.add_argument("--db-root", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--model", default="gpt-4o-2024-08-06", type=str)

    ap.add_argument("--packages", default="", type=str, help="Comma-separated packages; empty => auto calls∩db")
    ap.add_argument("--max-packages", type=int, default=0)

    ap.add_argument("--min-calls", type=int, default=0)
    ap.add_argument("--min-constraints", type=int, default=15)
    ap.add_argument("--max-constraints", type=int, default=250)

    ap.add_argument("--max-suffix-parts", type=int, default=5)
    ap.add_argument("--max-events-for-llm", type=int, default=250)

    ap.add_argument("--cache-dir", default=".llm_cache_span", type=str)
    ap.add_argument("--max-retries", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    """
    워크플로우는 아래와 같음:
    - calls load 이후 pkg_root별로 그룹핑
    - db_root에서 실제 pkg dir search
    - target pkg 선택(옵션 packages 또는 auto calls∩db)
    - 패키지별 run_one 실행 후 결과 저장
    """
    args = parse_args()
    calls_path = Path(args.calls)
    db_root = Path(args.db_root)
    out_path = Path(args.out)

    calls_by_pkg = load_calls_all(calls_path)
    pkg_dirs = find_package_dirs(db_root)
    db_pkgs = set(pkg_dirs.keys())

    if args.packages.strip():
        pkgs = [p.strip() for p in args.packages.split(",") if p.strip() and p.strip() in db_pkgs]
    else:
        observed = [(p, len(calls_by_pkg.get(p, []))) for p in calls_by_pkg.keys() if p in db_pkgs]
        observed.sort(key=lambda x: x[1], reverse=True)
        pkgs = [p for p, _ in observed]

    if args.max_packages and args.max_packages > 0:
        pkgs = pkgs[: args.max_packages]

    results: List[Dict[str, Any]] = []
    for pkg in pkgs:
        try:
            results.append(run_one(pkg, pkg_dirs[pkg], calls_by_pkg.get(pkg, []), args))
        except Exception as e:
            results.append({"package": pkg, "error": str(e), "stats": {"calls_total": len(calls_by_pkg.get(pkg, []))}})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".jsonl":
        write_jsonl(out_path, results)
    else:
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
