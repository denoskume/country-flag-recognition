from pathlib import Path


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


def test_uploaded_flag_placeholder_is_not_used_as_user_message():
    assert '"Flag image uploaded"' not in APP_SOURCE


def test_uploaded_flag_bytes_are_kept_with_user_chat_message():
    assert '"image_bytes": pending["file_bytes"]' in APP_SOURCE


def test_chat_renderer_supports_inline_uploaded_images():
    assert "def _render_chat_message(role: str, content: str, image_bytes: bytes | None = None)" in APP_SOURCE
    assert 'message.get("image_bytes")' in APP_SOURCE
    assert '<img class="fi-user-uploaded-image"' in APP_SOURCE


def test_uploaded_flag_preview_has_fixed_two_by_one_centimeter_size():
    assert 'width:2cm;height:1cm;' in APP_SOURCE
    assert 'object-fit:contain' in APP_SOURCE
    assert 'max-width:280px;max-height:190px' not in APP_SOURCE
