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
| `vacation_photo.png` | travel / photos / left in place | **binary** file, has to be classified by its name alone |
| `meeting_minutes_q2.txt` | work | |
| `scan_001.txt` | legal / housing | **misleading name**: a lease agreement |
| `course_certificate.txt` | education | |
| `todo.txt` | left in place (or personal) | **ambiguous**: the agent should leave it and say so |
| `.hidden_metadata` | — | hidden files are never listed |

Things to try during the run: reject a move with feedback (e.g. `no, put leases under housing`)
and check that the agent follows it, or press Ctrl-C partway through and make sure nothing else changes.
