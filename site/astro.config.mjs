import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

const SITE_URL = process.env.SITE_URL ?? 'https://sachncs.github.io';

export default defineConfig({
  site: SITE_URL,
  base: '/laker',
  trailingSlash: 'ignore',
  output: 'static',
  prefetch: {
    prefetchAll: true,
    defaultStrategy: 'viewport',
  },
  build: {
    inlineStylesheets: 'auto',
    assets: '_a',
  },
  image: {
    service: { entrypoint: 'astro/assets/services/sharp' },
  },
  integrations: [
    tailwind({
      applyBaseStyles: false,
      nesting: true,
    }),
  ],
  vite: {
    css: {
      devSourcemap: true,
    },
    build: {
      cssCodeSplit: true,
    },
  },
});