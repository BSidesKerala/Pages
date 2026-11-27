// Small-screen menu: the burger button opens and closes the link list.
(function () {
  var button = document.querySelector('.nav-toggle');
  var menu = document.getElementById('navMenu');
  if (!button || !menu) return;

  function setOpen(open) {
    menu.classList.toggle('open', open);
    button.setAttribute('aria-expanded', open ? 'true' : 'false');
    button.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  }

  button.addEventListener('click', function () {
    setOpen(!menu.classList.contains('open'));
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && menu.classList.contains('open')) {
      setOpen(false);
      button.focus();
    }
  });
})();
