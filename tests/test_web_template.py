from __future__ import annotations

from pathlib import Path


def test_index_template_includes_stream_cancel_and_multiline_sse_guards() -> None:
    html = Path("app/web/templates/index.html").read_text(encoding="utf-8")

    assert 'dataLines.join("\\n")' in html
    assert 'setStreamTerminalStatus("已取消")' in html
    assert 'sendBtnEl.disabled = busy;' in html
    assert 'runTestsBtnEl.disabled = busy;' in html
    assert 'saveSettingsBtnEl.disabled = busy;' in html
    assert 'if (eventName === "error")' in html
