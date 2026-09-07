# Vendored type

These four files are latin and latin-extended variable subsets of the two faces the **documentation
site** renders in. They are kept here, in the repository, so that no page load reaches a third-party
host — `effgen-docs/src/styles/globals.css` used to open with an `@import` from
`fonts.googleapis.com`, which meant every documentation page view made a request to Google before it
could draw text.

`effgen-docs/src/styles/globals.css` references them by relative path, so Vite emits and fingerprints
its own copy and the URLs stay correct under a base path.

| File | Family | Weights | Subset | Bytes |
|---|---|---|---|---|
| `inter-latin.woff2` | Inter | 100–900 | latin | 48,256 |
| `inter-latin-ext.woff2` | Inter | 100–900 | latin-ext | 85,068 |
| `jetbrains-mono-latin.woff2` | JetBrains Mono | 100–800 | latin | 40,404 |
| `jetbrains-mono-latin-ext.woff2` | JetBrains Mono | 100–800 | latin-ext | 15,196 |

A browser downloads only the subsets a page's text actually needs, so the usual cost is the two latin
files.

## Why the landing site has none of this

The landing site's `body` rule asks for `Inter`, and `app/layout.tsx` used to load Inter and Space
Grotesk through `next/font/google`. Those two facts never met: `next/font` publishes its faces under a
generated family name exposed as a CSS variable, and nothing in the landing stylesheet consumed that
variable — so the downloads happened on every page load and the page still rendered in the system
sans stack.

Removing `next/font/google` therefore changes nothing a visitor sees and drops two font downloads per
page. The landing site keeps rendering exactly as it does today, and no `@font-face` was added to it.
A visitor who has Inter installed locally still gets Inter, as they always did.

Should the landing site ever render in Inter after all, that is a design decision
rather than a build one: add an `@font-face` set to `app/globals.css` pointing at these same files,
with a relative path so webpack fingerprints them.

## Licences

Both families are licensed under the **SIL Open Font License, Version 1.1**.

- **Inter** — Copyright (c) 2016 The Inter Project Authors. <https://github.com/rsms/inter>
- **JetBrains Mono** — Copyright (c) 2020 The JetBrains Mono Project Authors.
  <https://github.com/JetBrains/JetBrainsMono>

The OFL permits redistribution of the font files, bundled with software, provided the copyright and
licence notice is retained — which is what this file is. The full licence text is at
<https://openfontlicense.org>.

## Replacing them

```bash
curl -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36' \
  'https://fonts.googleapis.com/css2?family=Inter:wght@100..900&family=JetBrains+Mono:wght@100..800&display=swap'
```

That returns one `@font-face` block per family per subset. Take the `latin` and `latin-ext` blocks,
download the `.woff2` each one names, and copy its `unicode-range` and `font-weight` into the
matching rules in `effgen-docs/src/styles/globals.css`. Nothing else needs to change.
