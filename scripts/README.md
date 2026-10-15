# Release checks

`smoke_test.py` exercises the public code with temporary synthetic data on CPU.
`audit_release.py` checks publication boundaries and reports only finding names
and file paths; it never prints suspected credential values.
