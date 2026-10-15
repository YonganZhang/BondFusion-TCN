# Numerical paper analysis

These modules reuse the original analysis algorithms without the figure style,
fonts, graphical assets or shared-skill imports. The numerical interval selector
in `_intervals.py` is unchanged. Start with
`python scripts/rebuild_paper_analysis.py --help`.

`_paper_data.py` resolves authorized local inputs and public aggregate fallbacks;
`_post_hoc.py` implements decision-rule, cost, Good-class, bootstrap and direct
design comparisons; the two builders assemble paper numbers and table fragments.
No manuscript or figure artwork is distributed here. Root release provenance
records the original source files and all changes.
