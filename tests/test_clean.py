from jobforge.clean import clean_description, find_boilerplate, html_to_lines, html_to_text


def test_unescapes_greenhouse_html_once():
    raw = "&lt;p&gt;Salary &amp;lt; $200k &amp;amp; equity&lt;/p&gt;"
    assert html_to_text(raw) == "Salary < $200k & equity"


def test_real_html_is_not_unescaped_twice():
    raw = "<p>Use &lt;div&gt; tags &amp; CSS</p>"
    assert html_to_text(raw) == "Use <div> tags & CSS"


def test_blocks_become_lines_and_whitespace_collapses():
    raw = "<h3>About  the\n role</h3><p>Build\tthings.&nbsp;Ship them.</p>line one<br>line two"
    assert html_to_lines(raw) == ["About the role", "Build things. Ship them.", "line one", "line two"]


def test_html_comments_are_dropped():
    assert html_to_text("<p>Hiring now<!-- internal note --></p>") == "Hiring now"


def test_inline_tags_do_not_split_sentences():
    assert html_to_text("<p>Experience with <strong>Python</strong> and <em>Go</em>.</p>") == \
        "Experience with Python and Go."


def test_list_items_get_bullets_even_when_nested_in_paragraphs():
    raw = "<ul><li>SQL</li><li>\n<p>Airflow</p></li></ul>"
    assert html_to_lines(raw) == ["- SQL", "- Airflow"]


def test_drops_scripts_and_greenhouse_boilerplate_divs():
    raw = ('<div class="content-intro"><p>We are a great company.</p></div>'
           "<p>Real job content.</p><script>track()</script>"
           '<div class="content-conclusion"><p>Equal opportunity employer.</p></div>')
    assert html_to_text(raw) == "Real job content."


def test_empty_input():
    assert html_to_lines("") == []
    assert html_to_text("   ") == ""


EEO = "We are an equal opportunity employer and value diversity at our company."


def _docs(n_with: int, n_without: int) -> list[list[str]]:
    return [["Build data pipelines", EEO]] * n_with + [["Build data pipelines"]] * n_without


def test_repeated_long_line_is_boilerplate():
    boilerplate = find_boilerplate(_docs(6, 2))
    assert clean_description(["Build data pipelines", EEO], boilerplate) == "Build data pipelines"


def test_short_lines_are_never_boilerplate():
    # "Build data pipelines" appears in every posting but is under min_words
    assert "build data pipelines" not in find_boilerplate(_docs(6, 2))


def test_line_below_share_threshold_is_kept():
    assert find_boilerplate(_docs(6, 10)) == set()  # 6 of 16 < 50%


def test_small_companies_need_min_docs():
    assert find_boilerplate(_docs(3, 0)) == set()  # 100% share, but only 3 postings


def test_boilerplate_match_ignores_bullets_and_case():
    boilerplate = find_boilerplate(_docs(6, 0))
    assert clean_description([f"- {EEO.upper()}", "Keep me"], boilerplate) == "Keep me"
