# LAKER — site

Marketing site for the [LAKER](https://github.com/sachncs/laker) project.
Built with [Astro](https://astro.build) + [Tailwind CSS](https://tailwindcss.com)
and deployed to GitHub Pages via a workflow at `.github/workflows/pages.yml`.

The site lives entirely in this directory. Nothing in the Python package
(`laker/`), the docs (`docs/`), or the repo root is rendered.

## Develop

```bash
cd site
npm install
npm run dev       # http://localhost:4321/laker
```

## Build

```bash
npm run build
# Output → site/dist/  (this is what gets deployed to GitHub Pages).
```

## Deploy

Push to `master` — `.github/workflows/pages.yml` builds `site/` and publishes
to GitHub Pages at <https://sachncs.github.io/laker/>.

## Layout

```
site/
├── astro.config.mjs        # site URL + base path (/laker), integrations
├── tailwind.config.mjs     # design tokens — colors, type scale, motion
├── package.json
├── public/                 # copied verbatim to dist/
│   ├── favicon.svg
│   ├── logo.svg
│   ├── og.svg              # social share image
│   └── robots.txt
└── src/
    ├── layouts/Base.astro  # HTML shell, fonts, JSON-LD
    ├── lib/site.ts         # canonical site config (title, links, paper)
    ├── styles/global.css   # design system: tokens, primitives, motion
    ├── components/         # one component per section
    └── pages/
        ├── index.astro     # the product page
        ├── 404.astro
        └── sitemap.xml.ts
```