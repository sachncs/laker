import type { APIRoute } from 'astro';
import { SITE } from '~/lib/site';

export const GET: APIRoute = ({ site }) => {
  const base = site?.toString().replace(/\/$/, '') ?? SITE.url;
  const items = [
    { loc: `${base}/`, changefreq: 'weekly', priority: '1.0' },
    { loc: `${base}/404`, changefreq: 'yearly', priority: '0.3' },
  ];
  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${items
  .map(
    (i) => `  <url>
    <loc>${i.loc}</loc>
    <changefreq>${i.changefreq}</changefreq>
    <priority>${i.priority}</priority>
  </url>`
  )
  .join('\n')}
</urlset>`;
  return new Response(body, {
    headers: { 'Content-Type': 'application/xml; charset=utf-8' },
  });
};