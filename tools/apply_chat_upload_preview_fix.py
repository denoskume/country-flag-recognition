from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "app.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Missing patch anchor: {label}")
    return text.replace(old, new, 1)


source = APP.read_text(encoding="utf-8")

source = replace_once(
    source,
    'def _render_chat_message(role: str, content: str) -> None:',
    'def _render_chat_message(role: str, content: str, image_bytes: bytes | None = None) -> None:',
    "chat renderer signature",
)

safe_line = '    safe = xml_escape(str(content or "")).replace("\\n", "<br/>")\n'
image_logic = '''    safe = xml_escape(str(content or "")).replace("\\n", "<br/>")
    image_html = ""
    if role == "user" and image_bytes:
        try:
            raw_image_bytes = bytes(image_bytes)
            mime_type = "image/png"
            if raw_image_bytes.startswith(b"\\xff\\xd8\\xff"):
                mime_type = "image/jpeg"
            elif raw_image_bytes.startswith(b"RIFF") and raw_image_bytes[8:12] == b"WEBP":
                mime_type = "image/webp"
            elif raw_image_bytes.startswith(b"GIF8"):
                mime_type = "image/gif"
            encoded_image = base64.b64encode(raw_image_bytes).decode("ascii")
            image_margin = ".55rem" if safe else "0"
            image_html = (
                f'<img class="fi-user-uploaded-image" '
                f'src="data:{mime_type};base64,{encoded_image}" alt="Uploaded flag" '
                f'style="display:block;max-width:280px;max-height:190px;width:auto;height:auto;'
                f'object-fit:contain;border-radius:12px;margin:0 0 {image_margin} 0;" />'
            )
        except Exception:
            image_html = ""
'''
source = replace_once(source, safe_line, image_logic, "inline image renderer")

source = replace_once(
    source,
    'f\'<div class="fi-user-message">{safe}</div>\'',
    'f\'<div class="fi-user-message">{image_html}{safe}</div>\'',
    "user message image markup",
)

source = replace_once(
    source,
    '_render_chat_message(message["role"], message["content"])',
    '_render_chat_message(message["role"], message["content"], message.get("image_bytes"))',
    "persisted chat rendering",
)

old_append = '''    if prompt_text:
        st.session_state.fi_messages.append(
            {"role": "user", "content": prompt_text}
        )
    elif prompt_files:
        st.session_state.fi_messages.append(
            {"role": "user", "content": "Flag image uploaded"}
        )
'''
new_append = '''    if prompt_text or prompt_files:
        st.session_state.fi_messages.append(
            {
                "role": "user",
                "content": prompt_text,
                "image_bytes": pending["file_bytes"],
            }
        )
'''
source = replace_once(source, old_append, new_append, "uploaded flag chat message")

APP.write_text(source, encoding="utf-8")
