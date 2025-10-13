# streamlit_app.py
# Canvas Course-wide HTML Rewrite (Test Instance)
# - Streamlit UI for admin-only dry-run + apply
# - Uses canvasapi + OpenAI Responses API
# - Preserves existing <iframe>s (domains unchanged) by freezing/restoring
# - Allows inline styles safely via nh3 filter_style_properties
# - DesignTools Mode (Preserve/Enhance/Replace) + free-form rewrite goals
# - Model Reference: Upload HTML (skeletonized), Upload Image (vision), or pick from a Model Course

import os
import re
import json
import base64
import hashlib
import urllib.parse
from typing import Optional, Tuple, List

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

def find_course_by_code(account, course_code: str) -> List:
    """Search by course_code; prefer exact code match if available."""
    courses = account.get_courses(search_term=course_code)
    matches = [c for c in courses if getattr(c, "course_code", "") == course_code]
    return matches if matches else list(courses)

def list_supported_items(course):
    """Enumerate Module Items and keep only Pages and Assignments."""
    supported = []
    for module in course.get_modules(include_items=True):
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

# ---- Robust upload readers ----

def read_text_upload(uploaded_file) -> str:
    """
    Return uploaded text file as str. Uses getvalue() to avoid pointer issues.
    Tries utf-8, falls back to latin-1 (lossy) so we never crash.
    """
    if uploaded_file is None:
        raise ValueError("No file provided")
    data = uploaded_file.getvalue()  # bytes (safe across reruns)
    if not data:
        raise ValueError("Empty file or no bytes read")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")

def image_to_data_url(file) -> Tuple[str, str]:
    """
    Accept a Streamlit UploadedFile and return (data_url, mime).
    Uses getvalue() instead of read() so we can preview and send to the model without re-upload.
    """
    if file is None:
        raise ValueError("No image provided")
    mime = file.type or "image/png"
    raw = file.getvalue()  # bytes
    b64 = base64.b64encode(raw).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    return data_url, mime

# ---- Model reference helpers ----

def html_to_skeleton(model_html: str, max_nodes: int = 1200) -> str:
    """
    Collapse a model HTML into a compact 'skeleton':
    - keep tags + ids/classes + data-*
    - drop most text and src/href values (avoid leaking course-specific links)
    - keep headings' text (shortened) to preserve hierarchy hints
    """
    soup = BeautifulSoup(model_html or "", "html.parser")

    # remove scripts/styles entirely
    for bad in soup(["script", "style"]):
        bad.decompose()

    count = 0
    for el in soup.find_all(True):
        count += 1
        if count > max_nodes:
            el.decompose()
            continue

        attrs = dict(el.attrs)
        new_attrs = {}
        for k, v in attrs.items():
            if k in ("id", "class") or k.startswith("data-") or k.startswith("aria-") or k == "role":
                new_attrs[k] = v
        el.attrs = new_attrs

        # remove inner text except short headings/labels
        if el.name not in ("h1","h2","h3","h4","h5","h6","summary","label"):
            el.string = None

    text = " ".join(str(soup).split())
    if len(text) > 120_000:
        text = text[:120_000] + " <!-- truncated -->"
    return text

def find_single_course_by_code(account, course_code: str):
    """Return list of courses matching code (UI will disambiguate)."""
    courses = account.get_courses(search_term=course_code)
    return list(courses)

def list_all_pages(course):
    """Return [(title_or_slug, url_slug)] for all pages."""
    pages = []
    for p in course.get_pages():
        pages.append((getattr(p, "title", p.url), p.url))
    return pages

def list_all_assignments(course):
    """Return [(name, id)] for all assignments."""
    items = []
    for a in course.get_assignments():
        items.append((getattr(a, "name", f"Assignment {a.id}"), a.id))
    return items

def fetch_model_item_html(course, kind: str, ident):
    """Fetch full HTML for the selected model item."""
    if kind == "Page":
        page = course.get_page(ident)  # ident = page.url (slug)
        return page.body or ""
    elif kind == "Assignment":
        asg = course.get_assignment(int(ident))  # ident = assignment id
        return asg.description or ""
    return ""

# ---------------------- OpenAI ----------------------

SYSTEM_PROMPT = (
    "You are an expert Canvas HTML editor. Preserve links, anchors/IDs, classes, and data-* attributes. "
    "Placeholders like ⟪IFRAME:n⟫ represent protected iframes—do not add, remove, or reorder them. "
    "Follow the policy. Return only HTML, no explanations."
)

DT_MODES = ["Preserve", "Enhance", "Replace"]

def openai_rewrite(
    user_request: str,
    html: str,
    dt_mode: str,
    model_html_skeleton: Optional[str] = None,
    model_image_data_url: Optional[str] = None,
) -> str:
    """
    Call OpenAI Responses API to rewrite the HTML.
    If model_image_data_url is present, send a multimodal input with an image.
    If model_html_skeleton is present, include it as a reference text block.
    """
    policy = {
        "design_tools_mode": dt_mode,
        "allow_inline_styles": True,
        "block_new_iframes": True,
        "reference_model": ("image" if model_image_data_url else "html" if model_html_skeleton else "none"),
        "reference_usage": "Match layout/sectioning/components; do not copy literal course-specific links or text."
    }

    text_blocks = [
        "Policy: " + json.dumps(policy, ensure_ascii=False),
        f"DesignTools Mode: {dt_mode}",
        "Rewrite goals: " + user_request,
        "HTML to rewrite follows:",
        html,
    ]
    if model_html_skeleton:
        text_blocks.insert(3, "Model HTML skeleton (follow structure/classes, not text):\n" + model_html_skeleton)

    if not model_image_data_url:
        prompt = "\n\n".join(text_blocks)
        resp = openai_client.responses.create(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    else:
        # Try multimodal; on failure, fall back to text-only
        try:
            content_parts = []
            content_parts.append({"type": "input_text", "text": "\n\n".join(text_blocks[:3])})
            if model_html_skeleton:
                content_parts.append({"type": "input_text", "text": text_blocks[3]})
            content_parts.append({"type": "input_image", "image_url": {"url": model_image_data_url}})
            content_parts.append({"type": "input_text", "text": "\n".join(text_blocks[-2:])})

            resp = openai_client.responses.create(
                model="gpt-4o",  # vision-capable model
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content_parts},
                ],
                temperature=0.2,
            )
        except Exception:
            prompt = "\n\n".join(text_blocks) + "\n\n(Note: image reference unavailable; proceed using textual model description if any.)"
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
        return resp.output[0].content[0].text  # fallback

# ---------------------- Diff & Apply ----------------------

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
    st.header("Rewrite Configuration")
    dt_mode = st.selectbox("DesignTools Mode", DT_MODES, index=1)
    user_request = st.text_area(
        "Rewrite goals (your instructions)",
        value="Normalize headings to start at h2; improve accessibility; preserve link destinations; "
              "convert or enhance DesignTools components as needed; keep existing iframes unchanged.",
        height=160,
        help="Describe exactly what to change across the course."
    )

    st.markdown("---")
    st.subheader("Model Reference (optional)")
    ref_kind = st.radio("Type", ["None", "Upload HTML", "Upload Image", "Model Course"], horizontal=True)

    model_html_skeleton = None
    model_image_data_url = None

    if ref_kind == "Upload HTML":
        uploaded_html = st.file_uploader("Upload model page HTML", type=["html", "htm"], accept_multiple_files=False)
        if uploaded_html is not None:
            try:
                raw_text = read_text_upload(uploaded_html)  # robust reader
                model_html_skeleton = html_to_skeleton(raw_text)
                with st.expander("Preview model HTML skeleton"):
                    st.code(model_html_skeleton[:4000])
            except Exception as e:
                st.error(f"Failed to parse model HTML: {e}")

    elif ref_kind == "Upload Image":
        uploaded_img = st.file_uploader("Upload model page image", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)
        if uploaded_img is not None:
            try:
                model_image_data_url, mime = image_to_data_url(uploaded_img)  # robust reader
                st.image(uploaded_img, caption=f"Model reference ({mime})", use_column_width=True)
                st.caption("The image will be passed to the model as a vision reference. If unsupported, we will fall back to text-only.")
            except Exception as e:
                st.error(f"Failed to process image: {e}")

    elif ref_kind == "Model Course":
        st.caption("Use a Page or Assignment from another Canvas course as the model.")
        mc_code = st.text_input("Model course code", key="model_course_code")
        if st.button("Find model course"):
            with st.spinner("Searching model course…"):
                st.session_state["model_courses"] = find_single_course_by_code(account, mc_code)

        model_courses = st.session_state.get("model_courses", [])
        if model_courses:
            m_idx = st.selectbox(
                "Select model course",
                options=list(range(len(model_courses))),
                format_func=lambda i: f"{model_courses[i].id} · {getattr(model_courses[i], 'course_code', '')} · {getattr(model_courses[i], 'name', '')}",
                key="model_course_idx"
            )
            model_course = model_courses[m_idx]

            kind = st.radio("Model item type", ["Page", "Assignment"], horizontal=True, key="model_item_kind")

            if kind == "Page":
                if "model_pages" not in st.session_state:
                    st.session_state["model_pages"] = list_all_pages(model_course)
                if st.button("Refresh model pages"):
                    st.session_state["model_pages"] = list_all_pages(model_course)
                pages = st.session_state.get("model_pages", [])
                if pages:
                    page_map = {f"{t}": slug for (t, slug) in pages}
                    sel_title = st.selectbox("Choose model page", options=list(page_map.keys()))
                    if sel_title:
                        try:
                            raw = fetch_model_item_html(model_course, "Page", page_map[sel_title])
                            model_html_skeleton = html_to_skeleton(raw)
                            with st.expander("Preview model HTML skeleton"):
                                st.code(model_html_skeleton[:4000])
                        except Exception as e:
                            st.error(f"Failed to fetch model page: {e}")

            else:  # Assignment
                if "model_asgs" not in st.session_state:
                    st.session_state["model_asgs"] = list_all_assignments(model_course)
                if st.button("Refresh model assignments"):
                    st.session_state["model_asgs"] = list_all_assignments(model_course)
                asgs = st.session_state.get("model_asgs", [])
                if asgs:
                    asg_map = {f"{name} (#{aid})": aid for (name, aid) in asgs}
                    sel_name = st.selectbox("Choose model assignment", options=list(asg_map.keys()))
                    if sel_name:
                        try:
                            raw = fetch_model_item_html(model_course, "Assignment", asg_map[sel_name])
                            model_html_skeleton = html_to_skeleton(raw)
                            with st.expander("Preview model HTML skeleton"):
                                st.code(model_html_skeleton[:4000])
                        except Exception as e:
                            st.error(f"Failed to fetch model assignment: {e}")

st.subheader("1) Pick Course by Code")
course_code = st.text_input("Course code", help="Exact course code preferred; we'll disambiguate if needed.")
if st.button("Search course"):
    with st.spinner("Searching…"):
        st.session_state["courses"] = find_course_by_code(account, course_code)

courses = st.session_state.get("courses", [])
if courses:
    idx = st.selectbox(
        "Select course",
        options=list(range(len(courses))),
        format_func=lambda i: f"{courses[i].id} · {getattr(courses[i], 'course_code', '')} · {getattr(courses[i], 'name', '')}",
    )
    course = courses[idx]

    st.subheader("2) Dry-run")
    if st.button("Collect items & simulate rewrite", disabled=(not user_request.strip())):
        if not user_request.strip():
            st.warning("Please enter rewrite goals first.")
        else:
            st.session_state["items"] = []
            st.session_state["drafts"] = {}
            with st.spinner("Fetching items…"):
                module_items = list_supported_items(course)

            progress = st.progress(0.0)
            total = max(len(module_items), 1)
            for n, (module, it) in enumerate(module_items, start=1):
                meta = fetch_item_html(course, it)
                original = meta["html"] or ""
                frozen, mapping, hosts = protect_iframes(original)

                # Call OpenAI with optional model reference
                rewritten = openai_rewrite(
                    user_request=user_request,
                    html=frozen,
                    dt_mode=dt_mode,
                    model_html_skeleton=model_html_skeleton,
                    model_image_data_url=model_image_data_url,
                )

                # Remove any new iframes the model tried to add, then restore originals
                rewritten_no_new_iframes = strip_new_iframes(rewritten)
                restored = restore_iframes(rewritten_no_new_iframes, mapping)

                # Sanitize final HTML (allow inline styles via property allowlist)
                sanitized = sanitize_html(restored)

                diff_html = html_diff(original, sanitized)
                key = f"{meta['kind']}:{meta['id']}"
                st.session_state["items"].append({
                    "key": key,
                    "title": meta.get("title") or meta.get("url"),
                    "kind": meta["kind"],
                    "module": getattr(module, "name", ""),
                    "item": it,
                    "original": original,
                    "draft": sanitized,
                    "approved": False,
                })
                st.session_state["drafts"][key] = {"diff": diff_html}
                progress.progress(n / total)
            st.success(f"Prepared {len(st.session_state['items'])} items.")

    items = st.session_state.get("items", [])
    if items:
        st.subheader("3) Review diffs & approve")
        approve_all = st.checkbox("Approve all")
        for _, rec in enumerate(items):
            with st.expander(f"[{rec['kind']}] {rec['title']} — {rec['module']}"):
                st.markdown("**Diff (proposed vs original)**  \n_Green = insertions, Red = deletions_", help="Generated by diff-match-patch")
                st.components.v1.html(st.session_state["drafts"][rec["key"]]["diff"], height=260, scrolling=True)
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
