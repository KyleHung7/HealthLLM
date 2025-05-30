document.addEventListener('DOMContentLoaded', function() {
    const navbarToggle = document.querySelector('.navbar__toggle');
    const navbarLinks = document.querySelector('.navbar__links');

    if (navbarToggle && navbarLinks) {
        navbarToggle.addEventListener('click', function() {
            navbarLinks.classList.toggle('active');
        });
    }
});