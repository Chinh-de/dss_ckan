/** @type {import('tailwindcss').Config} */

// Colours resolve to the CSS variables declared in src/index.css so the
// per-domain accent can be swapped at runtime via <html data-domain="…">.
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: v('bg'),
        surface: v('surface'),
        raised: v('raised'),
        ink: {
          DEFAULT: v('ink'),
          muted: v('ink-muted'),
          faint: v('ink-faint'),
        },
        line: {
          DEFAULT: 'rgb(var(--line) / 0.09)',
          strong: 'rgb(var(--line) / 0.18)',
        },
        accent: {
          DEFAULT: v('accent'),
          ink: v('accent-ink'),
          foreground: v('accent-ink'),
        },
        pos: v('pos'),
        neg: v('neg'),

        // Aliases kept for the older shadcn-style class names.
        background: v('bg'),
        foreground: v('ink'),
        card: { DEFAULT: v('surface'), foreground: v('ink') },
        popover: { DEFAULT: v('raised'), foreground: v('ink') },
        primary: { DEFAULT: v('accent'), foreground: v('accent-ink') },
        secondary: { DEFAULT: v('raised'), foreground: v('ink') },
        muted: { DEFAULT: v('raised'), foreground: v('ink-muted') },
        destructive: { DEFAULT: v('neg'), foreground: v('ink') },
        border: 'rgb(var(--line) / 0.09)',
        input: 'rgb(var(--line) / 0.12)',
        ring: v('accent'),
      },
      borderColor: {
        DEFAULT: 'rgb(var(--line) / 0.09)',
      },
      borderRadius: {
        lg: '10px',
        md: '7px',
        sm: '5px',
      },
      fontFamily: {
        sans: ['"Be Vietnam Pro"', 'system-ui', 'sans-serif'],
        display: ['"Bricolage Grotesque"', '"Be Vietnam Pro"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        lift: '0 28px 56px -28px rgb(4 3 2 / 0.85)',
        sheet: '-32px 0 64px -32px rgb(4 3 2 / 0.9)',
      },
      keyframes: {
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(10px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'sheet-in': {
          from: { opacity: '0', transform: 'translateX(32px)' },
          to: { opacity: '1', transform: 'translateX(0)' },
        },
        'pop-in': {
          from: { opacity: '0', transform: 'translateY(8px) scale(0.98)' },
          to: { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.25s ease-out both',
        'fade-up': 'fade-up 0.5s cubic-bezier(0.16, 1, 0.3, 1) both',
        'sheet-in': 'sheet-in 0.35s cubic-bezier(0.16, 1, 0.3, 1) both',
        'pop-in': 'pop-in 0.28s cubic-bezier(0.16, 1, 0.3, 1) both',
      },
    },
  },
  plugins: [],
};
