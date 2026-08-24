from yasbd import get_supported_langs, pysbd_adapter


def test_supported_langs_cover_app_asr_languages():
    langs = set(get_supported_langs())

    assert {"en", "ja", "zh", "ko"} <= langs


def test_adapter_keeps_pysbd_interface():
    seg = pysbd_adapter.Segmenter(language="ja", clean=False)

    parts = [p.strip() for p in seg.segment("今日はいい天気ですね。散歩に行きましょう。")]

    assert parts == ["今日はいい天気ですね。", "散歩に行きましょう。"]


def test_adapter_handles_english_abbreviations():
    seg = pysbd_adapter.Segmenter(language="en", clean=False)

    parts = [p.strip() for p in seg.segment("Mr. Smith paid 3.5 dollars. Dr. Lee disagreed.")]

    assert parts == ["Mr. Smith paid 3.5 dollars.", "Dr. Lee disagreed."]


def test_unsupported_language_is_detectable_for_fallback():
    # main._get_segmenter falls back to "en" for anything not in this list
    assert "xx" not in get_supported_langs()
