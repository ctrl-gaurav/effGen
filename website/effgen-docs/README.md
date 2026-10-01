# effgen-docs

The documentation site for [effGen](https://github.com/ctrl-gaurav/effGen) — 72 pages, built with
Vite and React and served under `/docs` on <https://www.effgen.org>.

This is one half of the [effgen.org](../README.md) repository. The landing site at the repository
root builds this bundle into its own `public/docs/` before exporting, so a normal release needs
only `npm run build` at the root.

## Running it on its own

```bash
npm install
npm run dev          # http://localhost:5173
npm run build        # search index, type-check, then bundle into ../public/docs
```

`npm run build` regenerates the search index first, so a page added since the last build is
searchable without a separate step.

## How a page is put together

The table of contents lives in exactly one file, `src/nav.ts`. Adding a page means adding an entry
there and one line to `PAGE_COMPONENTS` in `src/App.tsx`; the sidebar, the route table, the
breadcrumbs, the previous/next pair and the "see also" blocks all read `nav.ts`, so nothing else
holds a second copy of the list.

A page is written out of the primitives in `src/components/docs.ts`:

| | |
|---|---|
| `DocPage` | the page frame — title, lede, breadcrumbs, outline, previous/next |
| `CodeBlock`, `CodeTabs` | one sample, or one task done several ways |
| `Terminal` | a captured terminal session |
| `Figure` | a captured screenshot, with what produced it |
| `ParamTable`, `ApiTable` | options and parameters, matching `--help` exactly |
| `Callout`, `Steps`, `FeatureList`, `QuickLinks`, `SeeAlso` | the rest of the page furniture |
| `MermaidDiagram` | a diagram, in both colour modes, inside its own scroll box |

Every page carries a lede, a runnable example in the first screen, a table of the options it
describes, what happens when it fails, and a "see also" block.

## Facts and samples

Counts, versions, model ids and command names are not typed into a page. They come from
`data/effgen.json` at the repository root, which `scripts/gen_site_data.py` derives from the
installed framework; `vite.config.ts` aliases it as `@data` so there is one copy on disk.

Every code sample on either site has been run against the released framework. `test_all_snippets.py`
is what proves it — it extracts the samples out of the TSX, the JSON and the markdown, and runs
each under the right interpreter:

```bash
export PATH=/path/to/your/effgen/env/bin:$PATH
python test_all_snippets.py              # extract and run, from anywhere in the repository
python test_all_snippets.py --per-file   # what it found, per file
python test_all_snippets.py --list       # extract only
```

`snippet_policy.json` beside it records the samples that cannot run on an ordinary machine, keyed
by the sample's own text, each with a reason.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
