"""Deterministic verification.

`DONE` is an opinion. An assertion is a fact. Every check here is evaluated by
code against the live page, so a run can be judged without trusting a model's
summary of its own work.
"""

from __future__ import annotations

import fnmatch
import json
import re

from .observe import Observation

CHECK_TYPES = [
    "url_matches", "url_contains", "title_matches", "text_contains", "text_absent",
    "element_exists", "element_gone", "value_equals", "value_named", "field_shows", "checked",
    "count_at_least", "js",
]


def _fold(text: str | None) -> str:
    return " ".join((text or "").split()).casefold()


def _fields_named(observation: Observation, role: str | None, name: str):
    """Fields whose name (or label) starts with `name`; failing that, any that contain it.

    A prefix first because a combobox's accessible name carries its label *and* its current
    choice ("Where from? New York JFK"): the label is how the field is named, the rest is what
    it shows.
    """
    want = _fold(name)
    pool = [e for e in observation.elements if role is None or e.role == role]
    prefixed = [e for e in pool if _fold(e.name).startswith(want) or _fold(e.label).startswith(want)]
    return prefixed or [e for e in pool if want in _fold(e.name) or want in _fold(e.label)]


def _name_rest(element, name: str) -> str:
    """The element's name with the field-name part taken out: what the field *shows*."""
    have = _fold(element.name)
    want = _fold(name)
    if have.startswith(want):
        return have[len(want):].strip()
    at = have.find(want)
    return (have[:at] + " " + have[at + len(want):]).strip() if at >= 0 else have


def _fail(kind: str, detail: str) -> dict:
    return {"type": kind, "ok": False, "detail": detail}


def _ok(kind: str, detail: str) -> dict:
    return {"type": kind, "ok": True, "detail": detail}


def run(checks: list[dict], observation: Observation, *, allow_js: bool = False,
        eval_js=None) -> dict:
    results: list[dict] = []
    for check in checks or []:
        kind = str(check.get("type") or "").strip()
        try:
            results.append(_one(kind, check, observation, allow_js, eval_js))
        except Exception as exc:  # noqa: BLE001 - a bad check is a failed check
            results.append(_fail(kind or "unknown", f"check raised {type(exc).__name__}: {exc}"))
    return {
        "pass": bool(results) and all(result["ok"] for result in results),
        "checks": results,
        "url": observation.url,
    }


def _needle(check: dict, *keys: str) -> str:
    """The non-empty string a text-matching check has to be given.

    An absent or empty needle is a malformed check, not a match. `"" in s` is true for
    every string, so reading a missing key as an empty needle makes the check pass
    without looking at anything -- the one failure an assertion must not have, since a
    passing assertion is what overrules the model's own claim of success. A caller who
    misspells the key (`"value"` for `"text"`) got a pass rather than an error.
    """
    for key in keys:
        value = check.get(key)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def _one(kind: str, check: dict, observation: Observation, allow_js: bool, eval_js) -> dict:
    if kind == "url_matches":
        pattern = str(check.get("pattern") or check.get("url") or "")
        return (_ok if fnmatch.fnmatch(observation.url, pattern) else _fail)(
            kind, f"url {observation.url!r} vs pattern {pattern!r}")
    if kind == "url_contains":
        needle = _needle(check, "text")
        if not needle:
            return _fail(kind, "url_contains needs a non-empty 'text'")
        found = needle.lower() in observation.url.lower()
        return (_ok if found else _fail)(kind, f"url={observation.url!r} contains {needle!r}: {found}")
    if kind == "title_matches":
        pattern = _needle(check, "pattern", "text")
        if not pattern:
            return _fail(kind, "title_matches needs a non-empty 'pattern'")
        found = fnmatch.fnmatch(observation.title, pattern) or pattern.lower() in observation.title.lower()
        return (_ok if found else _fail)(kind, f"title={observation.title!r} vs {pattern!r}")
    if kind in {"text_contains", "text_absent"}:
        needle = _needle(check, "text")
        if not needle:
            return _fail(kind, f"{kind} needs a non-empty 'text'")
        # Page text is what is in the viewport; a results list below the fold is only in the element
        # names ("From 198 US dollars ..."). Measured on Google Flights: planner checks for
        # "US dollars" failed on a results page that was full of them. Secret fields' values are
        # masked in the observation already, so names and values add nothing a check may not see.
        haystack = "\n".join([observation.text or ""] + [
            f"{e.name} {e.value or ''}" for e in observation.elements if not getattr(e, "secret", False)])
        found = needle.lower() in haystack.lower()
        if check.get("regex"):
            found = bool(re.search(needle, haystack, re.IGNORECASE))
        want = kind == "text_contains"
        return (_ok if found == want else _fail)(
            kind, f"{needle!r} {'found' if found else 'not found'} in page text")
    if kind in {"element_exists", "element_gone"}:
        role = check.get("role")
        name = check.get("name") or check.get("text")
        ref = check.get("ref")
        if ref:
            found = ref in observation.by_ref
            detail = f"ref {ref} {'present' if found else 'absent'}"
        else:
            matches = observation.find(role, name)
            found = bool(matches)
            detail = (f"{len(matches)} match(es) for role={role!r} name={name!r}"
                      + (f" e.g. {matches[0].ref} {matches[0].name!r}" if matches else ""))
        want = kind == "element_exists"
        return (_ok if found == want else _fail)(kind, detail)
    if kind == "value_equals":
        ref = check.get("ref")
        element = observation.by_ref.get(ref or "")
        if element is None:
            return _fail(kind, f"ref {ref!r} is not on the page")
        expected = check.get("value")
        actual = element.value or element.current or ""
        found = str(actual) == str(expected)
        return (_ok if found else _fail)(kind, f"{ref} value={actual!r} vs expected={expected!r}")
    if kind in {"value_named", "field_shows"}:
        # `value_equals` needs a ref, which only exists once the page has been read -- so a check
        # written before the page it applies to (a plan) cannot use it. This finds the field by
        # role and name instead. A substring match by default, case-folded: a combobox that took
        # "JFK" shows "John F. Kennedy International (JFK)". `contains: false` asks for equality.
        name = _needle(check, "name")
        expected = _needle(check, "value")
        if not name or not expected:
            return _fail(kind, f"{kind} needs a non-empty 'name' and 'value'")
        role = check.get("role") or None
        contains = check.get("contains", True)
        if isinstance(contains, str):
            contains = contains.strip().lower() not in {"false", "0", "no", "off", ""}
        # `field_shows` is the form a plan should use: the field found by the start of its name,
        # and the expected text looked for in its value *or* in the rest of its name, since a
        # combobox commonly shows its choice there and not as a value ("Where from? New York
        # JFK" holds the code; its value says "New York"). `in_name` asks value_named the same.
        in_name = kind == "field_shows" or check.get("in_name") in {True, "true", "1", 1}
        matches = _fields_named(observation, role, name) if kind == "field_shows" \
            else observation.find(role, name)
        if not matches:
            return _fail(kind, f"no field with role={role!r} name={name!r}")
        want = " ".join(expected.split()).casefold()
        seen = []
        for element in matches:
            if element.secret:
                # A masked value is never read back, not even to compare it.
                seen.append(f"{element.ref} «hidden»")
                continue
            for actual in (element.value, element.current):
                have = " ".join((actual or "").split()).casefold()
                if have and (want in have if contains else want == have):
                    return _ok(kind, f"{element.ref} {element.name!r} value={actual!r} "
                                     f"{'contains' if contains else 'equals'} {expected!r}")
            if in_name:
                rest = _name_rest(element, name)
                if rest and (want in rest if contains else want == rest):
                    return _ok(kind, f"{element.ref} name={element.name!r} "
                                     f"{'contains' if contains else 'equals'} {expected!r}")
            seen.append(f"{element.ref} {(element.value or element.current or '')!r}"
                        + (f" name={element.name!r}" if in_name else ""))
        # A value asserted against a secret field is itself a secret; it is not echoed either.
        shown = "the expected value" if any(e.secret for e in matches) else repr(expected)
        return _fail(kind, f"{len(matches)} field(s) named {name!r}; none "
                           f"{'contains' if contains else 'equals'} {shown}: {', '.join(seen[:3])}")
    if kind == "checked":
        ref = check.get("ref")
        element = observation.by_ref.get(ref or "")
        if element is None:
            return _fail(kind, f"ref {ref!r} is not on the page")
        if "state" in check:
            raw = check["state"]
        elif "checked" in check:
            # The type is named `checked` and its argument is named `state`, so a caller
            # will guess `checked`. Reading it costs nothing, and a guess that lands is
            # worth more than being right about the key name.
            raw = check["checked"]
        else:
            # No default. `state` used to default to True, so a check that said nothing
            # about which state it wanted asserted "checked" -- and a caller writing
            # `"checked": false` got the opposite of what they wrote, as a pass.
            return _fail(kind, "checked needs 'state' (true or false)")
        if isinstance(raw, str):
            # `bool("false")` is True. A model writing JSON by hand sends strings, and
            # getting this wrong asserts the opposite of what was asked for.
            raw = raw.strip().lower() not in {"false", "0", "no", "off", ""}
        want = bool(raw)
        found = bool(element.checked) == want
        return (_ok if found else _fail)(kind, f"{ref} checked={element.checked} expected={want}")
    if kind == "count_at_least":
        role = check.get("role")
        name = check.get("name") or check.get("text")
        minimum = int(check.get("min") or 1)
        count = len(observation.find(role, name))
        found = count >= minimum
        return (_ok if found else _fail)(kind, f"{count} match(es) >= {minimum}: {found}")
    if kind == "js":
        if not allow_js:
            return _fail(kind, "JS checks are disabled; set JEVMCP_ALLOW_JS=1 to enable")
        expression = str(check.get("expr") or "")
        if not expression or eval_js is None:
            return _fail(kind, "js check needs 'expr'")
        value = eval_js(expression)
        return (_ok if value else _fail)(kind, f"{expression} -> {json.dumps(value)[:120]}")
    return _fail(kind or "unknown", f"unknown check type; supported: {', '.join(CHECK_TYPES)}")
