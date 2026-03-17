# Research Site Design System (Premium Standard)

This document formalizes the "Premium" UI/UX standard for the research site, characterized by glassmorphism, modern typography, and dynamic backgrounds.

## 1. Design Tokens (CSS Variables)

All pages should implement or inherit the following tokens in `:root`:

| Token | Value | Description |
|---|---|---|
| `--glass-bg` | `rgba(255, 255, 255, 0.7)` | Primary background for cards/panels |
| `--glass-border` | `rgba(255, 255, 255, 0.3)` | Subtle edge for glass elements |
| `--premium-shadow` | `0 20px 40px rgba(0, 0, 0, 0.05)`| Soft, deep elevation |
| `--font-main` | `'Space Grotesk'` | Core body typography |
| `--font-accent` | `'Outfit'` | Headings and UI accents |
| `--brand-gradient` | `linear-gradient(135deg, #005f73 0%, #0a9396 100%)` | Primary brand action color |

## 2. Global Aesthetics

### Backdrop
The standard background is a triple-gradient providing depth without distraction:
```css
linear-gradient(120deg, #f0f4f8 0%, #d9e2ec 100%),
radial-gradient(at 0% 0%, rgba(0, 95, 115, 0.05) 0%, transparent 50%),
radial-gradient(at 100% 0%, rgba(202, 103, 2, 0.05) 0%, transparent 50%);
```

### Glassmorphism
Panels must use `backdrop-filter` to enforce the glass aesthetic:
```css
background: var(--glass-bg);
backdrop-filter: blur(12px);
-webkit-backdrop-filter: blur(12px);
border: 1px solid var(--glass-border);
border-radius: 24px;
```

## 3. Typography Hierarchy

- **Page Titles**: `h1`, 3.5rem, `Outfit` (700), Gradient-clipped text.
- **Section Headers**: `h2`, 1.8rem, `Outfit` (600), Solid ink.
- **Body Text**: `p`, 1.1rem, `Space Grotesk` (400), High legibility.

## 4. UI Components

- **Executive Summary**: Always anchored with a leftmost brand border (`6px`).
- **Metric Cards**: Dynamic number rendering with bold, large typography (`2.5rem`).
- **Interactive Pills**: Rounded (`999px`) with subtle borders, inheriting theme colors.

## 5. Implementation Anchoring
To ensure a run adheres to this standard, the `manifest.yaml` should include:
```yaml
ui_standard: "premium"
```
If missing, the site fallback to "standard" (legacy) styling.
