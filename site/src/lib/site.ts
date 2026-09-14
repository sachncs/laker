export const SITE = {
  name: 'LAKER',
  title: 'LAKER — Attention Kernel Regression, at the speed of science',
  description:
    'A PyTorch library for regularised attention kernel regression with a learned preconditioner. Scales to 100k+ samples with near size-independent convergence for spectrum cartography and beyond.',
  url: 'https://sachncs.github.io/laker/',
  repo: 'https://github.com/sachncs/laker',
  repoShort: 'sachncs/laker',
  pypi: 'https://pypi.org/project/laker/',
  paper: 'https://arxiv.org/abs/2604.25138',
  docs: 'https://sachncs.github.io/laker/',
  version: '0.5.0',
  paperMeta: {
    title: 'Accelerating Regularized Attention Kernel Regression for Spectrum Cartography',
    authors: 'Tao & Tan (2026)',
    venue: 'arXiv:2604.25138',
  },
  social: {
    github: 'https://github.com/sachncs/laker',
    issues: 'https://github.com/sachncs/laker/issues',
    discussions: 'https://github.com/sachncs/laker/discussions',
  },
} as const;

export const NAV = [
  { label: 'Overview', href: '#overview' },
  { label: 'Architecture', href: '#architecture' },
  { label: 'Benchmarks', href: '#benchmarks' },
  { label: 'Quick start', href: '#quick-start' },
] as const;