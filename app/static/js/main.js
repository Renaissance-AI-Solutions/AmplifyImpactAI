// Document ready function
$(document).ready(function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Handle form submissions
    $('form').on('submit', function(e) {
        var $form = $(this);
        if ($form.find('.is-invalid').length) {
            e.preventDefault();
            return false;
        }
    });

    // Handle file uploads
    $('input[type="file"]').on('change', function() {
        var fileName = $(this).val().split('\\').pop();
        $(this).next('.custom-file-label').html(fileName);
    });

    // Handle comment approval
    $('.approve-comment').on('click', function(e) {
        e.preventDefault();
        var $btn = $(this);
        var commentId = $btn.data('comment-id');
        
        $btn.prop('disabled', true);
        $btn.find('.spinner-border').show();

        $.ajax({
            url: '/engagement/comments/' + commentId + '/approve',
            method: 'POST',
            success: function(response) {
                showNotification('Comment posted successfully!', 'success');
                setTimeout(function() {
                    window.location.reload();
                }, 1000);
            },
            error: function(xhr) {
                showNotification('Error posting comment: ' + xhr.responseText, 'danger');
                $btn.prop('disabled', false);
                $btn.find('.spinner-border').hide();
            }
        });
    });

    // Handle comment rejection
    $('.reject-comment').on('click', function(e) {
        e.preventDefault();
        var $btn = $(this);
        var commentId = $btn.data('comment-id');
        
        if (confirm('Are you sure you want to reject this comment?')) {
            $.ajax({
                url: '/engagement/comments/' + commentId + '/reject',
                method: 'POST',
                success: function(response) {
                    showNotification('Comment rejected successfully!', 'success');
                    setTimeout(function() {
                        window.location.reload();
                    }, 1000);
                },
                error: function(xhr) {
                    showNotification('Error rejecting comment: ' + xhr.responseText, 'danger');
                }
            });
        }
    });

    // Handle post scheduling
    $('.schedule-post').on('click', function(e) {
        e.preventDefault();
        var $btn = $(this);
        var postId = $btn.data('post-id');
        
        $btn.prop('disabled', true);
        $btn.find('.spinner-border').show();

        $.ajax({
            url: '/content-studio/posts/' + postId + '/schedule',
            method: 'POST',
            data: {
                scheduled_time: $('#scheduled_time').val(),
                target_account_id: $('#target_account_id').val()
            },
            success: function(response) {
                showNotification('Post scheduled successfully!', 'success');
                setTimeout(function() {
                    window.location.reload();
                }, 1000);
            },
            error: function(xhr) {
                showNotification('Error scheduling post: ' + xhr.responseText, 'danger');
                $btn.prop('disabled', false);
                $btn.find('.spinner-border').hide();
            }
        });
    });

    // Handle AI content generation
    $('.generate-ai-content').on('click', function(e) {
        e.preventDefault();
        var $btn = $(this);
        
        $btn.prop('disabled', true);
        $btn.find('.spinner-border').show();

        $.ajax({
            url: '/content-studio/generate-ai-content',
            method: 'POST',
            data: {
                topic: $('#topic').val(),
                tone: $('#tone').val(),
                style: $('#style').val()
            },
            success: function(response) {
                $('#generated_content').val(response.content);
                showNotification('AI content generated successfully!', 'success');
            },
            error: function(xhr) {
                showNotification('Error generating AI content: ' + xhr.responseText, 'danger');
            },
            complete: function() {
                $btn.prop('disabled', false);
                $btn.find('.spinner-border').hide();
            }
        });
    });

    // Handle knowledge base search
    $('.search-knowledge').on('click', function(e) {
        e.preventDefault();
        var $btn = $(this);
        
        $btn.prop('disabled', true);
        $btn.find('.spinner-border').show();

        $.ajax({
            url: '/knowledge-base/search',
            method: 'POST',
            data: {
                query: $('#search_query').val()
            },
            success: function(response) {
                displaySearchResults(response.results);
                showNotification('Search completed successfully!', 'success');
            },
            error: function(xhr) {
                showNotification('Error searching knowledge base: ' + xhr.responseText, 'danger');
            },
            complete: function() {
                $btn.prop('disabled', false);
                $btn.find('.spinner-border').hide();
            }
        });
    });
});

// Show notification function
function showNotification(message, type) {
    var toast = document.createElement('div');
    toast.className = 'toast';
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');

    var toastHeader = document.createElement('div');
    toastHeader.className = 'toast-header';
    
    var toastBody = document.createElement('div');
    toastBody.className = 'toast-body';
    toastBody.textContent = message;

    toast.appendChild(toastHeader);
    toast.appendChild(toastBody);
    document.body.appendChild(toast);

    var bsToast = new bootstrap.Toast(toast);
    bsToast.show();

    setTimeout(function() {
        bsToast.hide();
        toast.remove();
    }, 5000);
}

// Display search results function
function displaySearchResults(results) {
    var resultsContainer = $('#search_results');
    resultsContainer.empty();

    if (results.length === 0) {
        resultsContainer.append('<p>No results found.</p>');
        return;
    }

    results.forEach(function(result) {
        var resultItem = $('<div>').addClass('card mb-2');
        var resultBody = $('<div>').addClass('card-body');
        
        var title = $('<h6>').addClass('card-title').text(result.title);
        var content = $('<p>').addClass('card-text').text(result.content);
        var score = $('<small>').addClass('text-muted').text('Relevance: ' + (result.score * 100).toFixed(1) + '%');

        resultBody.append(title, content, score);
        resultItem.append(resultBody);
        resultsContainer.append(resultItem);
    });
}

// Handle file upload progress
function handleFileUploadProgress(event) {
    var progress = Math.round((event.loaded / event.total) * 100);
    $('#upload-progress').css('width', progress + '%');
    $('#upload-progress').text(progress + '%');
}

// Handle file upload complete
function handleFileUploadComplete(xhr) {
    var response = JSON.parse(xhr.responseText);
    if (response.success) {
        showNotification('File uploaded successfully!', 'success');
        setTimeout(function() {
            window.location.reload();
        }, 1000);
    } else {
        showNotification('Error uploading file: ' + response.message, 'danger');
    }
}
