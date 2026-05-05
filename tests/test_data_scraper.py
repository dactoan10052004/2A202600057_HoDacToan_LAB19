from src.data_scraper import _split_template_params, clean_wikitext


def test_clean_wikitext_removes_links_and_templates() -> None:
    text = "[[Larry Page]] and [[Sergey Brin|Sergey M. Brin]] {{nowrap|Google}}"
    assert clean_wikitext(text) == "Larry Page and Sergey M. Brin Google"


def test_split_template_params_extracts_infobox_fields() -> None:
    template = """{{Infobox company
| name = Example
| founder = [[Alice Example]]<br/>[[Bob Example]]
| founded = {{start date and age|2015|12|11}}
}}"""
    fields = _split_template_params(template)
    assert fields["founder"] == "Alice Example; Bob Example"
    assert fields["founded"] == "2015"

