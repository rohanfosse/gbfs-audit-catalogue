# DOI lookup for paper/references.bib (cited-only entries)

**Date:** 2026-05-14
**Tool:** WebSearch (Parallel APIs unavailable; PARALLEL_API_KEY not set)
**Context:** ScholarEval citation dimension was scored 4.0 — 32/72 bib entries had DOIs. Filtering to *cited* entries only, 15 journal/conference papers were missing DOIs; 14 were standards/tools/web-docs (DOI not applicable).

## Verified DOIs to add

| Bib key | DOI | Source |
|---|---|---|
| Wang1996Quality | 10.1080/07421222.1996.11518099 | tandfonline.com (JMIS 12(4):5-33) |
| Pipino2002Assessment | 10.1145/505248.506010 | dl.acm.org (CACM 45(4):211-218) |
| Pucher2010Infrastructure | 10.1016/j.ypmed.2009.07.028 | sciencedirect (Preventive Medicine 50:S106-S125) |
| Khatri2010DataGovernance | 10.1145/1629175.1629210 | dl.acm.org (CACM 53(1):148-152) |
| Naumann2014DataProfiling | 10.1145/2590989.2590995 | dl.acm.org (SIGMOD Record 42(4):40-49) |
| Janssen2012OpenData | 10.1080/10580530.2012.716740 | tandfonline.com (IS Management 29(4):258-268) |
| Stonebraker2005OneSize | 10.1109/ICDE.2005.1 | dl.acm.org / IEEE Xplore (ICDE 2005:2-11) |
| Lin2018BSSGraph | 10.1016/j.trc.2018.10.011 | sciencedirect (Transp Res Part C 97:258-276) |
| Buehler2017BikeShare | 10.2105/AJPH.2016.303546 | ajph.aphapublications.org (AJPH 107(2):281-287) |
| Eren2020Review | 10.1016/j.scs.2019.101882 | sciencedirect (Sustainable Cities & Society 54:101882) |
| Hughes2007QuickCheck | 10.1007/978-3-540-69611-7_1 | link.springer.com (PADL 2007, LNCS 4354:1-32) |

## Metadata correction (year/volume/pages wrong)

| Bib key | Currently in bib | Verified correct |
|---|---|---|
| Medard2017Rebalancing | year=2017, vol=64, pages=218-233 | **year=2016, vol=55, pages=22-39, DOI=10.1016/j.jtrangeo.2016.07.003** |

Source: dial.uclouvain.be / liser.elsevierpure.com / ideas.repec.org (all consistent)

## ISBN instead of DOI (book without DOI)

| Bib key | ISBN | Source |
|---|---|---|
| SebastianColeman2013 | 978-0-12-397033-6 | Elsevier shop / Amazon (Morgan Kaufmann, 2013) |

## No DOI exists (genuinely)

- **Antrim2013Conveyal** — ITS America 23rd Annual Meeting conference paper, not assigned a DOI
- **NABSA2024StateOfIndustry** — trade-association report
- **ISO25012, ISO25024, ISO19157** — ISO standards (use ISO numbering instead)
- **INSPIRE2007** — EU directive
- **DCAT2020, DCATAP2021, W3CDQV** — W3C / EC recommendations (use URL)
- **Mobi2024, OMF2023, MobilityData2024Validator** — open-standard specs / validators (URL)
- **FrictionlessData2023, GreatExpectations2024, Pandera2024, Croissant2024** — software / tooling docs (URL)

## Audit finding flagged separately

- **Wegier2023OSMQuality** is in references.bib but appears to be **non-existent / hallucinated**: no paper matching the cited title+authors+venue was found via WebSearch. Authors "Wegier, Reichholf" don't appear to have published this. → Entry is **unused** in manuscript.tex (no `\cite{Wegier2023OSMQuality}`), so it doesn't affect the compiled bibliography, but should be deleted from references.bib to avoid future accidental citation.
- 35 bib entries are entirely unused — they don't appear in the rendered bibliography (unsrt) but bloat the file. Worth pruning.
