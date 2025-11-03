import re
from typing import List, Dict

import pytest


def _polluted_entries() -> List[Dict]:
    """Return a list of intentionally polluted YouTube result dicts to simulate tool output before sanitization."""
    return [
        {
            "url": '<div class="wrap">https://www.youtube.com/watch?v=AbCd123_9" rel="nofollow">Watch</div>',
            "title": "  <b>How to Replace Sensor</b>  ",
            "thumbnail": "https://img.youtube.com/vi/AbCd123_9/default/default.jpg",
        },
        {
            # Short youtu.be link, extra trailing query params & whitespace
            "url": "  https://youtu.be/ZzYYxX_77?t=15  ",
            "title": "Quick   Diagnostic   Guide",
            # Missing thumbnail forces fallback
        },
        {
            # Invalid (no youtube) should be skipped by sanitization final pass
            "url": "https://example.com/not-a-youtube-link",
            "title": "Irrelevant Link",
            "thumbnail": "https://example.com/image.jpg",
        },
    ]


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    """Stub out llm.invoke to avoid external API calls when creating voice summary."""
    import tools.RAG_tools as RAG_tools

    class DummyLLM:
        def invoke(self, prompt: str):
            return type("Resp", (), {"content": "Short summary for voice."})()

    monkeypatch.setattr(RAG_tools, "llm", DummyLLM())


def test_youtube_sanitization_and_render_data():
    import tools.RAG_tools as RAG_tools
    # Access original function behind @tool decorator
    fmt_func = getattr(RAG_tools.format_diagnostic_results, "func", RAG_tools.format_diagnostic_results)

    polluted = _polluted_entries()

    # relevance_score=1 triggers RAG formatting branch, but we supply a simple rag_answer
    result = fmt_func(
        question="How do I replace the O2 sensor?",
        rag_answer="Basic steps to replace the sensor.",
        web_results=[],
        youtube_results=polluted,
        relevance_score=1,
    )

    assert "formatted_response" in result, "Expected structured formatted_response in result"
    fr = result["formatted_response"]
    assert "text_output" in fr and "voice_output" in fr

    videos = fr["text_output"]["youtube_videos"]
    # First invalid (non-youtube) entry should be filtered out, leaving 2 valid YouTube entries
    assert len(videos) == 2, f"Expected 2 sanitized YouTube videos, got {len(videos)}: {videos}"

    for v in videos:
        # URL should be clean and match expected YouTube patterns
        assert re.match(r"https?://(www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+", v["url"]) or re.match(
            r"https?://(www\.)?youtu\.be/[A-Za-z0-9_-]+", v["url"]
        )
        # Title should not contain HTML tags or excessive internal spacing
        assert "<" not in v["title"] and ">" not in v["title"], f"Title still contains HTML: {v['title']}"
        assert "  " not in v["title"], f"Title not space-collapsed: {v['title']}"
        # Thumbnail should never be the degraded default/default.jpg variant
        assert "default/default.jpg" not in v["thumbnail"], f"Degraded thumbnail not replaced: {v['thumbnail']}"
        # Should end with a known YouTube static image pattern
        assert re.search(r"img\.youtube\.com/vi/.+/(mqdefault|hqdefault)\.jpg", v["thumbnail"])
        # video_id should be present and alphanumeric/underscore
        assert re.match(r"[A-Za-z0-9_-]+", v["video_id"])

    # The main diagnostic content should not contain raw youtube watch URLs anymore
    content = fr["text_output"]["content"]
    assert "youtube.com/watch" not in content, "Content still contains raw YouTube watch URL"
    assert "youtu.be/ZzYYxX_77" not in content, "Short link still present in content"

    # Voice summary should be short (enforced by stub)
    voice = fr["voice_output"].strip()
    assert len(voice.split()) <= 50, "Voice summary longer than expected"

def test_youtube_empty_list_when_no_valid_entries(monkeypatch):
    import tools.RAG_tools as RAG_tools
    fmt_func = getattr(RAG_tools.format_diagnostic_results, "func", RAG_tools.format_diagnostic_results)

    # All invalid entries
    polluted = [
        {"url": "https://example.com/abc", "title": "Example", "thumbnail": "https://example.com/i.jpg"},
        {"url": "not even a url", "title": "Broken", "thumbnail": None},
    ]

    result = fmt_func(
        question="Generic?",
        rag_answer="Minimal.",
        web_results=[],
        youtube_results=polluted,
        relevance_score=1,
    )

    videos = result["formatted_response"]["text_output"]["youtube_videos"]
    assert videos == [], f"Expected empty list for invalid YouTube entries, got: {videos}"
