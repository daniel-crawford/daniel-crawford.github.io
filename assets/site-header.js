(async function loadSharedHeader() {
  const mount = document.getElementById('site-header');
  if (!mount) return;

  try {
    const response = await fetch('assets/header.html', { cache: 'no-cache' });
    if (!response.ok) throw new Error(`Failed to load header: ${response.status}`);

    mount.innerHTML = await response.text();

    const page = document.body.dataset.page;
    if (!page) return;

    const activeLink = mount.querySelector(`[data-page="${page}"]`);
    if (activeLink) {
      activeLink.classList.add('font-semibold', 'underline');
    }
  } catch (error) {
    console.error(error);
  }
})();
