# EyeLab technical wiki

Bilingual (EN/IT) wiki covering the vision pipeline, the validation campaign, the
XREAL port and the decision record. Open `index.html` in a browser (equations and
fonts load from the web; without a connection the TeX source is shown instead).

- Edit the sources in `src/` (`pages/*.html` one file per part, `wiki.css`, `wiki.js`).
  Every text exists twice, in `.l-en` and `.l-it` elements; numbers in `table.data`
  cells use a decimal point and are converted for Italian at runtime.
- Rebuild: `python docs/wiki/build_wiki.py` (standard library only) → `index.html`.
- Images in `img/` are frames from validation campaign 1 (Surface webcam, 15–16 Oct 2026).
- The wiki summarises the repository documents; when they disagree, the repository
  documents (AGENTS.md precedence list) win and the wiki is corrected.
- Public-safe: no content from `.docs/`; access-route details are summarised neutrally.
