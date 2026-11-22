# Demo data

Real sample files for running the agent end-to-end against a real LLM.

```bash
uv run python demo/demo.py reset      # fresh copy: demo/sample -> demo/inbox (git-ignored)
uv run file-organizer demo/inbox      # organize it, approving/rejecting each change
uv run python demo/demo.py check      # compare the result with demo/expected.json
```

`demo/sample/` is the pristine source and is never touched by the agent; reset as often as you like.

## What the sample exercises

| File | Expected | Why it's there |
|---|---|---|
| `invoice_2024_031.txt`, `bank_statement_march.csv` | finance | easy cases, different formats |
| `finance/receipt_laptop.txt` | finance (stays) | existing category dir must be **reused** and its files left alone |
| `tax_return_2023.txt` | finance / taxes | ~8.6 KB, exercises `read_file` **truncation** |
| `notes.txt` | health | **misleading name**: a doctor's prescription |
| `lab_results.md` | health | markdown table |
| `IMG_2291.txt` | travel | **misleading name**: a boarding pass, not an image |
| `booking_confirmation.eml`, `itinerary.json` | travel | email and JSON formats |
| `vacation_photo.png` | travel / photos / unsorted / left in place | **image**: classified by its name and metadata (EXIF) |
| `meeting_minutes_q2.txt` | work | |
| `scan_001.txt` | legal / housing | **misleading name**: a lease agreement |
| `course_certificate.txt` | education | |
| `todo.txt` | unsorted (or personal / left in place) | **ambiguous**: the agent should move it to `unsorted/` and say so |
| `electricity_bill_feb_2024.pdf` | finance | **PDF** with a table |
| `cv_leo_gozman.docx` | work / career | **DOCX** |
| `policy_documents.doc` | insurance / finance / legal | legacy **Word 97 DOC**, ~11K characters: the key clause is only in the *middle* sample |
| `budget_2024.xlsx` | finance | **XLSX**, two sheets, 14 columns |
| `onboarding.ppt` | work | legacy **PowerPoint** with speaker notes |
| `IMG_4410.jpg` | travel / photos | **JPEG with EXIF**: camera, date, description and GPS (Lisbon) |
| `Downloads/` | — | **not a category**: its files (also in the nested `old stuff/`) must be sorted into top-level categories; the emptied folders are removed with the approved moves that empty them |
| `Downloads/scan0003.pdf` | health | **misleading name**: a vaccination certificate |
| `Downloads/attachment (1).docx` | housing / legal | spaces and parentheses in the name; a landlord's deposit letter |
| `Downloads/team_offsite.xls` | work | legacy **Excel XLS**, dates, two sheets |
| `Downloads/old stuff/Document1.doc` | education | **misleading name**, nested 2 levels: a university transcript |
| `Downloads/old stuff/presentation.ppt` | education | lecture slides with notes |
| `Downloads/old stuff/export.xlsx` | health | **misleading name**: a blood-pressure log |
| `.hidden_metadata` | — | hidden files are never listed |

The office documents were written for this demo and converted with LibreOffice (pandoc for
the DOCX files); the photo is drawn with Pillow, its EXIF data set by hand. Reading PDF, DOC,
XLS, PPT and EXIF needs the `documents` extra (`uv sync --extra documents`); without it those
files show metadata only.

Things to try during the run: reject a move with feedback (e.g. `no, put leases under housing`)
and check that the agent follows it, or press Ctrl-C partway through and make sure nothing else changes.
