# streamlit_app.py
# Canvas Course-wide HTML Rewrite (Test Instance)
# - Streamlit UI for admin-only dry-run + apply
# - Uses canvasapi + OpenAI Responses API
# - Preserves existing <iframe>s (domains unchanged) by freezing/restoring
# - Allows inline styles safely via nh3 filter_style_properties
# - DesignTools Mode (Preserve/Enhance/Replace) + simple presets

import os
import json
import hashlib
import urllib.parse

import streamlit as st
from canvasapi import Canvas
from openai import OpenAI
from bs4 import BeautifulSoup
from diff_match_patch import diff_match_patch
import nh3

# ---------------------- Config & Clients ----------------------

SECRETS = st.secrets if hasattr(st, "secrets") else os.environ

CANVAS_BASE_URL = SECRETS["CANVAS_BASE_URL"]
CANVAS_ACCOUNT_ID = int(SECRETS["CANVAS_ACCOUNT_ID"])
CANVAS_ADMIN_TOKEN = SECRETS["CANVAS_ADMIN_TOKEN"]
OPENAI_API_KEY = SECRETS["OPENAI_API_KEY"]

st.set_page_config(page_title="Canvas Course-wide HTML Rewrite (Test)", layout="wide")

canvas = Canvas(CANVAS_BASE_URL, CANVAS_ADMIN_TOKEN)
account = canvas.get_account(CANVAS_ACCOUNT_ID)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------- Policy / Sanitizer ----------------------

ALLOWED_TAGS = [
    "a", "abbr", "b", "blockquote", "br", "code", "div", "em",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img",
    "li", "ol", "p", "pre", "s", "section", "small", "span", "strong",
    "sub", "sup", "table", "thead", "tbody", "tfoot", "tr", "th", "td",
    "u", "ul", "details", "summary", "iframe"
]

ATTRIBUTES = {
    "*": ["id", "class", "style", "title", "role", "aria-*", "data-*"],
    "a": ["href", "target", "rel"],
    "img": ["src", "alt", "width", "height"],
    "iframe": ["src", "width", "height", "allow", "allowfullscreen"],
    "th": ["scope"],
    "td": ["colspan", "rowspan"],
}

# Conservative but useful set of CSS properties for inline styles
ALLOWED_CSS_PROPERTIES = {
    # layout
    "display", "visibility", "float", "clear", "overflow", "overflow-x", "overflow-y",
    "position", "top", "right", "bottom", "left", "z-index",
    "box-sizing",
    # spacing & size
    "margin", "margin-top", "margin-right", "margin-bottom", "margin-left",
    "padding", "padding-top", "padding-right", "padding-bottom", "padding-left",
    "width", "min-width", "max-width", "height", "min-height", "max-height",
    # borders & backgrounds
    "border", "border-top", "border-right", "border-bottom", "border-left",
    "border-width", "border-style", "border-color", "border-radius",
    "background", "background-color", "background-image", "background-position",
    "background-size", "background-repeat",
    # text & fonts
    "color", "font", "font-family", "font-size", "font-weight", "font-style",
    "font-variant", "line-height", "letter-spacing", "text-align",
    "text-decoration", "text-transform", "white-space", "word-break",
    # flex & grid (common for themes)
    "flex", "flex-direction", "flex-wrap", "flex-grow", "flex-shrink", "flex-basis",
    "justify-content", "align-items", "align-content", "gap", "row-gap", "column-gap",
    "grid", "grid-template", "grid-template-columns", "grid-template-rows", "grid-column", "grid-row",
    # effects
    "opacity", "box-shadow",
}

def sanitize_html(html: str) -> str:
    """Sanitize HTML, allowing inline styles for a curated set of properties."""
    return nh3.clean(
        html,
        tags=set(ALLOWED_TAGS),
        attributes={k: set(v) for k, v in ATTRIBUTES.items()},
        filter_style_properties=set(ALLOWED_CSS_PROPERTIES),
        link_rel=None,  # don't force rel=... unless you want to enforce noopener here
    )

# ---------------------- Helpers ----------------------

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def find_course_by_code(account, course_code: str):
    """Search by course_code; prefer exact code match if available."""
    courses = account.get_courses(search_term=course_code)
    matches = [c for c in courses if getattr(c, "course_code", "") == course_code]
    return matches if matches else list(courses)

def list_supported_items(course):
    """Enumerate Module Items and keep only Pages and Assignments."""
    supported = []
    for module in course.get_modules(include_items=True):
        # Some Canvas instances may not populate 'items' without an explicit call:
        items = list(module.get_module_items()) if not getattr(module, "items", None) else module.items
        for it in items:
            if it.type in {"Page", "Assignment"}:
                supported.append((module, it))
    return supported

def fetch_item_html(course, item):
    """Fetch the HTML-bearing body for a supported item."""
    if item.type == "Page":
        page = course.get_page(item.page_url)
        return {"kind": "Page", "id": page.page_id, "url": page.url, "title": page.title, "html": page.body or ""}
    elif item.type == "Assignment":
        asg = course.get_assignment(item.content_id)
        return {"kind": "Assignment", "id": asg.id, "title": asg.name, "html": asg.description or ""}

# Freeze/restore existing iframes so their domains remain unchanged
PLACEHOLDER_FMT = "⟪IFRAME:{i}⟫"

def protect_iframes(html: str):
    """Replace existing iframes with placeholders and return (frozen_html, mapping, hosts)."""
    soup = BeautifulSoup(html or "", "html.parser")
    mapping, hosts = {}, set()
    i = 1
    for tag in soup.find_all("iframe"):
        src = (tag.get("src") or "").strip()
        host = urllib.parse.urlparse(src).hostname or ""
        if host:
            hosts.add(host.lower())
        placeholder = PLACEHOLDER_FMT.format(i=i)
        mapping[placeholder] = str(tag)
        tag.replace_with(placeholder)
        i += 1
    return str(soup), mapping, hosts

def restore_iframes(html: str, mapping: dict) -> str:
    """Replace placeholders with the original <iframe> markup."""
    for ph, original in mapping.items():
        html = html.replace(ph, original)
    return html

def strip_new_iframes(html: str) -> str:
    """Remove any iframes the model added (we only want to restore originals)."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup.find_all("iframe"):
        tag.decompose()
    return str(soup)

SYSTEM_PROMPT = (
    "You are an expert Canvas HTML editor. Preserve links, anchors/IDs, classes, and data-* attributes. "
    "Placeholders like ⟪IFRAME:n⟫ represent protected iframes—do not add, remove, or reorder them. "
    "Follow the policy. Return only HTML, no explanations."
)

PRESETS = {
    "Accessibility Boost (Enhance)": (
        "Improve accessibility while preserving meaning. Start headings at h2. "
        "Fix tables (th scope). Add aria-labels to unlabeled interactive controls. "
        "Do not change link URLs. Use alt=\"\" if missing."
    ),
    "Theme Refresh (Enhance)": (
        "Apply the {THEME_NAME} theme. Replace legacy theme classes with theme-{THEME_NAME}. "
        "Prefer class-based styling; keep necessary inline styles."
    ),
    "Migrate DT → Native (Replace)": (
        "Replace DesignTools accordions and tabs with native HTML. "
        "Use <details><summary> for accordions and structured <section> with headings for tabs. "
        "Keep IDs/anchors; update references if changed."
    ),
    "Heading Normalize (Preserve)": (
        "Normalize headings to start at h2; remove empty <p> and extra <br>. "
        "Do not change DesignTools features."
    ),
    "Link Hygiene (Enhance)": (
        "Convert bare URLs to descriptive links when context is clear; otherwise leave as-is. "
        "Add rel=\"noopener\" to links with target=\"_blank\"."
    ),
}

DT_MODES = ["Preserve", "Enhance", "Replace"]

def openai_rewrite(user_request: str, html: str, dt_mode: str) -> str:
    """Call OpenAI Responses API to rewrite the HTML according to request & policy."""
    policy = {
        "design_tools_mode": dt_mode,
        "allow_inline_styles": True,
        "block_new_iframes": True,
    }

    prompt = "\n\n".join([
        "Policy: " + json.dumps(policy, ensure_ascii=False),
        f"DesignTools Mode: {dt_mode}",
        "Rewrite goals: " + user_request,
        "HTML to rewrite follows:",
        html,
    ])

    resp = openai_client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    # Robust extraction across SDK variants
    try:
        return resp.output_text  # newer SDKs
    except AttributeError:
        return resp.output[0].content[0].text  # fallback

def html_diff(old: str, new: str) -> str:
    dmp = diff_match_patch()
    d = dmp.diff_main(old or "", new or "")
    dmp.diff_cleanupSemantic(d)
    return dmp.diff_prettyHtml(d)

def apply_update(course, item, new_html: str):
    if item.type == "Page":
        course.get_page(item.page_url).edit(wiki_page={"body": new_html})
    elif item.type == "Assignment":
        course.get_assignment(item.content_id).edit(assignment={"description": new_html})

# ---------------------- UI ----------------------

st.title("Canvas Course-wide HTML Rewrite (Test Instance)")

with st.sidebar:
    st.header("Configuration")
    dt_mode = st.selectbox("DesignTools Mode", DT_MODES, index=1)
    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    preset_vars = {}
    if "Theme Refresh" in preset_name:
        preset_vars["THEME_NAME"] = st.text_input("Theme name", value="institution")
    user_request = PRESETS[preset_name].format(**preset_vars) if preset_vars else PRESETS[preset_name]
    st.caption("Effective rewrite goals:")
    st.code(user_request)

st.subheader("1) Pick Course by Code")
course_code = st.text_input("Course code", help="Exact course code preferred; we'll disambiguate if needed.")
if st.button("Search course"):
    with st.spinner("Searching…"):
        st.session_state.courses = find_course_by_code(account, course_code)

courses = st.session_state.get("courses", [])
if courses:
    idx = st.selectbox(
        "Select course",
        options=list(range(len(courses))),
        format_func=lambda i: f"{courses[i].id} · {getattr(courses[i], 'course_code', '')} · {getattr(courses[i], 'name', '')}",
    )
    course = courses[idx]

    st.subheader("2) Dry-run")
    if st.button("Collect items & simulate rewrite"):
        st.session_state.items = []
        st.session_state.drafts = {}
        with st.spinner("Fetching items…"):
            items = list_supported_items(course)

        progress = st.progress(0.0)
        total = max(len(items), 1)
        for n, (module, it) in enumerate(items, start=1):
            meta = fetch_item_html(course, it)
            original = meta["html"] or ""
            frozen, mapping, hosts = protect_iframes(original)

            # Call OpenAI
            rewritten = openai_rewrite(user_request, frozen, dt_mode)

            # Remove any new iframes the model tried to add, then restore originals
            rewritten_no_new_iframes = strip_new_iframes(rewritten)
            restored = restore_iframes(rewritten_no_new_iframes, mapping)

            # Sanitize final HTML (allow inline styles via property allowlist)
            sanitized = sanitize_html(restored)

            diff_html = html_diff(original, sanitized)
            key = f"{meta['kind']}:{meta['id']}"
            st.session_state.items.append({
                "key": key,
                "title": meta.get("title") or meta.get("url"),
                "kind": meta["kind"],
                "module": getattr(module, "name", ""),
                "item": it,
                "original": original,
                "draft": sanitized,
                "approved": False,
            })
            st.session_state.drafts[key] = {"diff": diff_html}
            progress.progress(n / total)
        st.success(f"Prepared {len(st.session_state.items)} items.")

    items = st.session_state.get("items", [])
    if items:
        st.subheader("3) Review diffs & approve")
        approve_all = st.checkbox("Approve all")
        for _, rec in enumerate(items):
            with st.expander(f"[{rec['kind']}] {rec['title']} — {rec['module']}"):
                st.markdown("**Diff (proposed vs original)**  \n_Green = insertions, Red = deletions_", help="Generated by diff-match-patch")
                st.components.v1.html(st.session_state.drafts[rec["key"]]["diff"], height=260, scrolling=True)
                approved = st.checkbox("Approve this item", key=f"approve_{rec['key']}", value=approve_all)
                rec["approved"] = approved
                with st.expander("Show HTML (original)"):
                    st.code(rec["original"][:2000])
                with st.expander("Show HTML (proposed)"):
                    st.code(rec["draft"][:2000])

        st.write("")
        if st.button("Apply approved changes"):
            applied, failed = 0, 0
            with st.spinner("Applying to Canvas…"):
                for rec in items:
                    if not rec["approved"]:
                        continue
                    try:
                        apply_update(course, rec["item"], rec["draft"])
                        applied += 1
                    except Exception as e:
                        failed += 1
                        st.error(f"Failed: {rec['title']}: {e}")
            st.success(f"Applied {applied} item(s); {failed} failed.")
else:
    st.info("Search for a course above to begin.")
