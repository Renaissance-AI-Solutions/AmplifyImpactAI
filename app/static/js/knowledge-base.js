document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('#uploadModal form');
    if (!form) return;
    const submitButton = form.querySelector('button[type="submit"], input[type="submit"]');
    if (!submitButton) return;
    form.addEventListener('submit', function(e) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
    });
});
