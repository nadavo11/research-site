(() => {
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
})();
