# Working on the documentation site

## Prerequisites

Node 20 and npm 11, and the repository's dependencies installed at both levels.

## Run it

```bash
npm install
npm run dev
# http://localhost:5173
```

The dev server serves the site at the root. In production it is mounted at `/docs`, which
`vite.config.ts` sets through `base`; router paths come from `import.meta.env.BASE_URL`, so no
path is written twice.

## Build it

```bash
npm run build     # search index, type-check, then bundle into ../public/docs
```

Building the whole site from the repository root is usually what you want — it puts this bundle
where the landing site's static export will pick it up:

```bash
cd .. && npm run build
```

## Publishing

Nothing is published from this directory. The landing site's export in `out/` carries the built
documentation under `out/docs`, and that export is what Netlify publishes and what the framework
repository mirrors to GitHub Pages.
