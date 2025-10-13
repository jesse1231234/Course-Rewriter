import os, hashlib, urllib.parse
import streamlit as st
from canvasapi import Canvas
from openai import OpenAI
from bs4 import BeautifulSoup
from diff_match_patch import diff_match_patch
import nh3
from nh3 import CssSanitizer
import json

# ---------------------- Config & Clients ----------------------
SECRETS = st.secrets if hasattr(st, 'secrets') else os.environ
CANVAS_BASE_URL = SECRETS["CANVAS_BASE_URL"]
CANVAS_ACCOUNT_ID = int(SECRETS["CANVAS_ACCOUNT_ID"])
CANVAS_ADMIN_TOKEN = SECRETS["CANVAS_ADMIN_TOKEN"]
OPENAI_API_KEY = SECRETS["OPENAI_API_KEY"]

st.set_page_config(page_title="Canvas Course-wide HTML Rewrite (Test)", layout="wide")

canvas = Canvas(CANVAS_BASE_URL, CANVAS_ADMIN_TOKEN)
account = canvas.get_account(CANVAS_ACCOUNT_ID)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------- Helpers ----------------------
PLACEHOLDER_FMT = "⟪IFRAME:{i}⟫"

ALLOWED_TAGS = [
    "a","abbr","b","blockquote","br","code","div","em","h1","h2","h3","h4","h5","h6","hr","i","img",
    "li","ol","p","pre","s","section","small","span","strong","sub","sup","table","thead","tbody","tfoot",
    "tr","th","td","u","ul","details","summary","iframe"
]
ATTRIBUTES = {
    "*": ["id","class","style","title","role","aria-*","data-*"],
    "a": ["href","target","rel"],
    "img": ["src","alt","width","height"],
    "iframe": ["src","width","height","allow","allowfullscreen"],
    "th": ["scope"],
    "td": ["colspan","rowspan"],
}
CSS = CssSanitizer()

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def find_course_by_code(account, course_code: str):
    courses = account.get_courses(search_term=course_code)
    # Prefer exact code match when available
    matches = [c for c in courses if getattr(c, 'course_code', '') == course_code]
    return matches if matches else list(courses)

def list_supported_items(course):
    supported = []
    for module in course.get_modules(include_items=True):
        items = list(module.get_module_items()) if not getattr(module, 'items', None) else module.items
        for it in items:
            if it.type in {"Page", "Assignment"}:
                supported.append((module, it))
    return supported

def fetch_item_html(course, item):
    if item.type == "Page":
        page = course.get_page(item.page_url)
        return {"kind": "Page", "id": page.page_id, "url": page.url, "title": page.title, "html": page.body}
    elif item.type == "Assignment":
        asg = course.get_assignment(item.content_id)
        return {"kind": "Assignment", "id": asg.id, "title": asg.name, "html": asg.description or ""}

# Iframe freeze/restore so domains remain unchanged

def protect_iframes(html: str):
    soup = BeautifulSoup(html or '', 'html.parser')
    mapping, hosts = {}, set()
    i = 1
    for tag in soup.find_all('iframe'):
        src = (tag.get('src') or '').strip()
        host = urllib.parse.urlparse(src).hostname or ''
        if host:
            hosts.add(host.lower())
        placeholder = PLACEHOLDER_FMT.format(i=i)
        mapping[placeholder] = str(tag)
        tag.replace_with(placeholder)
        i += 1
    return str(soup), mapping, hosts

def restore_iframes(html: str, mapping: dict) -> str:
    for ph, original in mapping.items():
        html = html.replace(ph, original)
    return html

SYSTEM_PROMPT = (
    "You are an expert Canvas HTML editor. Preserve links, anchors/IDs, classes, data-* attributes. "
    "Placeholders like ⟪IFRAME:n⟫ represent protected iframes—do not add or remove them, although they can be repositioned within the page. "
    "Follow the policy. Return only HTML, no explanations."
    "Reformat the HTML using DesignPLUS styling. Do not change the content of the page, only the design. Apply Headings and accordions where appropriate. Use Colorado State University branding colors. Apply consistent DesignPLUS structure with a header, content blocks, headings, lists, callouts for notes or reminders, and accessible image formatting. The focus is on styling, structure, and accessibility — not changing the content."
)

PRESETS = {
    "Accessibility Boost (Enhance)": "Improve accessibility while preserving meaning. Start headings at h2. Fix tables (th scope). Add aria-labels to unlabeled interactive controls. Do not change link URLs. Use alt=\"\" if missing.",
    "Theme Refresh (Enhance)": "Apply the {THEME_NAME} theme. Replace legacy theme classes with theme-{THEME_NAME}. Prefer class-based styling; keep necessary inline styles.",
    "Migrate DT → Native (Replace)": "Replace DesignTools accordions and tabs with native HTML. Use <details><summary> for accordions and structured <section> with headings for tabs. Keep IDs/anchors; update references if changed.",
    "Heading Normalize (Preserve)": "Normalize headings to start at h2; remove empty <p> and extra <br>. Do not change DT features.",
    "Link Hygiene (Enhance)": "Convert bare URLs to descriptive links when context is clear; otherwise leave as-is. Add rel=\"noopener\" to target=\"_blank\" links."
}

DT_MODES = ["Preserve", "Enhance", "Replace"]

# Sanitize allowing inline styles; allow only original iframe hosts
class _IframeFilter(nh3.HtmlSanitizer):
    def __init__(self, allowed_hosts):
        self.allowed_hosts = set(allowed_hosts or [])
    def allow_url(self, url: str) -> bool:
        host = (urllib.parse.urlparse(url).hostname or '').lower()
        return (host in self.allowed_hosts) or any(host.endswith('.'+h) for h in self.allowed_hosts)

def sanitize_html(html: str, allowed_hosts: set) -> str:
    return nh3.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ATTRIBUTES,
        css_sanitizer=CSS,
        url_sanitizer=_IframeFilter(allowed_hosts),
    )

def openai_rewrite(user_request: str, html: str, dt_mode: str) -> str:
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
    try:
        return resp.output_text  # newer SDKs
    except AttributeError:
        return resp.output[0].content[0].text  # fallback for older SDKs

# Diff

def html_diff(old: str, new: str) -> str:
    dmp = diff_match_patch()
    d = dmp.diff_main(old or '', new or '')
    dmp.diff_cleanupSemantic(d)
    return dmp.diff_prettyHtml(d)

# Apply update

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
    st.code(user_request)

st.subheader("1) Pick Course by Code")
course_code = st.text_input("Course code", help="Exact course code preferred; we'll disambiguate if needed.")
if st.button("Search course"):
    with st.spinner("Searching…"):
        st.session_state.courses = find_course_by_code(account, course_code)

courses = st.session_state.get('courses', [])
if courses:
    idx = st.selectbox("Select course", options=list(range(len(courses))),
                       format_func=lambda i: f"{courses[i].id} · {getattr(courses[i], 'course_code', '')} · {getattr(courses[i], 'name', '')}")
    course = courses[idx]

    st.subheader("2) Dry‑run")
    if st.button("Collect items & simulate rewrite"):
        st.session_state.items = []
        st.session_state.drafts = {}
        with st.spinner("Fetching items…"):
            items = list_supported_items(course)
        progress = st.progress(0)
        for n, (module, it) in enumerate(items, start=1):
            meta = fetch_item_html(course, it)
            original = meta["html"] or ""
            frozen, mapping, hosts = protect_iframes(original)
            rewritten = openai_rewrite(user_request, frozen, dt_mode)
            restored = restore_iframes(rewritten, mapping)
            sanitized = sanitize_html(restored, hosts)
            diff_html = html_diff(original, sanitized)
            key = f"{meta['kind']}:{meta['id']}"
            st.session_state.items.append({
                "key": key,
                "title": meta.get("title") or meta.get("url"),
                "kind": meta["kind"],
                "module": getattr(module, 'name', ''),
                "item": it,
                "original": original,
                "draft": sanitized,
                "approved": False,
                "hosts": hosts,
            })
            st.session_state.drafts[key] = {
                "diff": diff_html,
            }
            progress.progress(n/len(items))
        st.success(f"Prepared {len(st.session_state.items)} items.")

    items = st.session_state.get('items', [])
    if items:
        st.subheader("3) Review diffs & approve")
        approve_all = st.checkbox("Approve all")
        for idx, rec in enumerate(items):
            with st.expander(f"[{rec['kind']}] {rec['title']} — {rec['module']}"):
                st.markdown("**Diff (proposed vs original)**", help="Green inserts, red deletions")
                st.components.v1.html(st.session_state.drafts[rec['key']]['diff'], height=240, scrolling=True)
                approved = st.checkbox("Approve this item", key=f"approve_{rec['key']}", value=approve_all)
                rec['approved'] = approved
                with st.expander("Show HTML (original)"):
                    st.code(rec['original'][:2000])
                with st.expander("Show HTML (proposed)"):
                    st.code(rec['draft'][:2000])
        st.write("")
        if st.button("Apply approved changes"):
            applied, failed = 0, 0
            with st.spinner("Applying to Canvas…"):
                for rec in items:
                    if not rec['approved']:
                        continue
                    try:
                        apply_update(course, rec['item'], rec['draft'])
                        applied += 1
                    except Exception as e:
                        failed += 1
                        st.error(f"Failed: {rec['title']}: {e}")
            st.success(f"Applied {applied} item(s); {failed} failed.")
else:
    st.info("Search for a course above to begin.")
