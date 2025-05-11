const navbarToggle = document.querySelector('.navbar__toggle');
const navbarLinks = document.querySelector('.navbar__links');

navbarToggle.addEventListener('click', () => {
  navbarLinks.classList.toggle('active');
});
