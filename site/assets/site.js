(() => {
  const storageKey = 'research-site-theme';
  const root = document.documentElement;
  const themeQuery = typeof window.matchMedia === 'function'
    ? window.matchMedia('(prefers-color-scheme: dark)')
    : null;

  function readStoredTheme() {
    try {
      const stored = window.localStorage.getItem(storageKey);
      return stored === 'dark' || stored === 'light' ? stored : null;
    } catch {
      return null;
    }
  }

  function writeStoredTheme(theme) {
    try {
      window.localStorage.setItem(storageKey, theme);
    } catch {
      // Ignore storage failures and keep the active theme for this page view.
    }
  }

  function preferredTheme() {
    return themeQuery?.matches ? 'dark' : 'light';
  }

  function buttonLabel(theme) {
    return theme === 'dark' ? 'Light mode' : 'Dark mode';
  }

  function ensureThemeToggle() {
    if (document.querySelector('[data-theme-toggle]')) return;
    const host = document.querySelector('.container') || document.querySelector('main') || document.body;
    if (!host) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'page-tools';

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'theme-toggle';
    button.setAttribute('data-theme-toggle', '');
    button.setAttribute('aria-label', 'Toggle color theme');
    button.setAttribute('aria-pressed', 'false');

    const label = document.createElement('span');
    label.setAttribute('data-theme-toggle-label', '');
    label.textContent = 'Dark mode';

    button.appendChild(label);
    wrapper.appendChild(button);
    host.insertBefore(wrapper, host.firstChild);
  }

  function applyTheme(theme) {
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
      button.setAttribute('aria-pressed', String(theme === 'dark'));
      button.setAttribute('title', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
      const label = button.querySelector('[data-theme-toggle-label]');
      if (label) {
        label.textContent = buttonLabel(theme);
      } else {
        button.textContent = buttonLabel(theme);
      }
    });
  }

  ensureThemeToggle();
  applyTheme(root.dataset.theme === 'dark' || root.dataset.theme === 'light' ? root.dataset.theme : readStoredTheme() || preferredTheme());

  document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      const nextTheme = root.dataset.theme === 'dark' ? 'light' : 'dark';
      writeStoredTheme(nextTheme);
      applyTheme(nextTheme);
    });
  });

  if (themeQuery) {
    const handleThemeChange = (event) => {
      if (readStoredTheme()) return;
      applyTheme(event.matches ? 'dark' : 'light');
    };
    if (typeof themeQuery.addEventListener === 'function') {
      themeQuery.addEventListener('change', handleThemeChange);
    } else if (typeof themeQuery.addListener === 'function') {
      themeQuery.addListener(handleThemeChange);
    }
  }

  const detailButtons = document.querySelectorAll('[data-toggle-target]');
  detailButtons.forEach((btn) => {
    const targetId = btn.getAttribute('data-toggle-target');
    const target = document.getElementById(targetId);
    if (!target) return;
    btn.addEventListener('click', () => {
      const isOpen = target.classList.toggle('open');
      btn.textContent = isOpen ? 'Hide details' : 'Show details';
    });
  });

  const dialog = document.getElementById('imageDialog');
  const dialogImage = document.getElementById('imageDialogImg');
  if (dialog && dialogImage) {
    document.querySelectorAll('[data-zoom-src]').forEach((img) => {
      img.addEventListener('click', () => {
        dialogImage.src = img.getAttribute('data-zoom-src');
        dialogImage.alt = img.alt || 'Expanded preview';
        dialog.showModal();
      });
    });
    dialog.addEventListener('click', (event) => {
      const rect = dialog.getBoundingClientRect();
      const inside = rect.top <= event.clientY && event.clientY <= rect.bottom && rect.left <= event.clientX && event.clientX <= rect.right;
      if (!inside) dialog.close();
    });
  }

  const filter = document.getElementById('expFilter');
  if (filter) {
    filter.addEventListener('change', () => {
      const value = filter.value;
      document.querySelectorAll('[data-exp-card]').forEach((card) => {
        const matches = value === 'all' || card.getAttribute('data-domain') === value;
        card.style.display = matches ? '' : 'none';
      });
    });
  }

  document.querySelectorAll('[data-sort-gallery]').forEach((sel) => {
    const galleryId = sel.getAttribute('data-sort-gallery');
    const gallery = document.getElementById(galleryId);
    if (!gallery) return;
    const original = Array.from(gallery.children);
    sel.addEventListener('change', () => {
      const items = Array.from(gallery.children);
      let sorted;
      if (sel.value === 'asc') {
        sorted = items.slice().sort((a, b) => (parseFloat(a.dataset.score) || 0) - (parseFloat(b.dataset.score) || 0));
      } else if (sel.value === 'desc') {
        sorted = items.slice().sort((a, b) => (parseFloat(b.dataset.score) || 0) - (parseFloat(a.dataset.score) || 0));
      } else {
        sorted = original;
      }
      sorted.forEach((el) => gallery.appendChild(el));
    });
  });
})();
