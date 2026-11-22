# Legacy Office fixtures

Real files for `tests/test_legacy_office.py`, written by LibreOffice 24 (headless), with
content made up for these tests:

| File | Made from | Exercises |
|---|---|---|
| `word97_contract.doc` | an HTML page, `soffice --convert-to 'doc:MS Word 97'` | Word 97 piece table with UTF-16 text (umlauts, €), a hyperlink field, a table, a title in the summary information |
| `excel97_budget.xls` | a flat ODS file, `soffice --convert-to xls` | two sheets, date cells, more than 5 rows |

LibreOffice can't write Word 6/95, and its `.ppt` files are ~450 KB (embedded extras); the
tests build those, and damaged files, with a minimal OLE2 writer and record builder instead.
