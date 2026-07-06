/**
 * Lightbox Gallery System for Research Visualizations
 * 
 * Sets up accessibility indicators, intercepts image wrappers clicks,
 * manages overlay transitions, and handles key binding events.
 */

document.addEventListener('DOMContentLoaded', () => {
    const plotWrappers = Array.from(document.querySelectorAll('.plot-wrapper'));
    const lightboxModal = document.getElementById('lightbox-modal');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxCaption = document.getElementById('lightbox-caption');
    const closeBtn = document.querySelector('.lightbox-close');
    const prevBtn = document.querySelector('.lightbox-prev');
    const nextBtn = document.querySelector('.lightbox-next');
    
    if (!lightboxModal || !lightboxImg || !lightboxCaption || plotWrappers.length === 0) {
        return;
    }

    // Extract paths and captions metadata from static page elements
    const plotsData = plotWrappers.map(wrapper => {
        const img = wrapper.querySelector('.plot-img');
        const card = wrapper.closest('.plot-card');
        const caption = card ? card.querySelector('.plot-caption').innerHTML : '';
        return {
            src: img ? img.src : '',
            alt: img ? img.alt : '',
            caption: caption
        };
    });
    
    let currentIndex = 0;
    let lastActiveElement = null;
    
    function openLightbox(index) {
        lastActiveElement = document.activeElement;
        currentIndex = index;
        updateLightboxContent();
        lightboxModal.classList.add('active');
        lightboxModal.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden'; // Stop viewport scroll
        if (closeBtn) closeBtn.focus();
    }
    
    function closeLightbox() {
        lightboxModal.classList.remove('active');
        lightboxModal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = ''; // Re-enable viewport scroll
        if (lastActiveElement) {
            lastActiveElement.focus();
        }
    }
    
    function updateLightboxContent() {
        const data = plotsData[currentIndex];
        if (data) {
            lightboxImg.src = data.src;
            lightboxImg.alt = data.alt;
            lightboxCaption.innerHTML = data.caption;
        }
    }
    
    function showNext() {
        currentIndex = (currentIndex + 1) % plotsData.length;
        updateLightboxContent();
    }
    
    function showPrev() {
        currentIndex = (currentIndex - 1 + plotsData.length) % plotsData.length;
        updateLightboxContent();
    }
    
    // Bind click and keys for triggers
    plotWrappers.forEach((wrapper, index) => {
        wrapper.setAttribute('tabindex', '0');
        wrapper.setAttribute('role', 'button');
        const img = wrapper.querySelector('.plot-img');
        const altText = img ? img.alt : 'Plot Visualization';
        wrapper.setAttribute('aria-label', `Expand figure: ${altText}`);
        
        wrapper.addEventListener('click', () => {
            openLightbox(index);
        });
        
        wrapper.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openLightbox(index);
            }
        });
    });
    
    if (closeBtn) closeBtn.addEventListener('click', closeLightbox);
    
    // Close on clicking outside the container overlay
    lightboxModal.addEventListener('click', (e) => {
        if (e.target === lightboxModal) {
            closeLightbox();
        }
    });
    
    if (prevBtn) {
        prevBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            showPrev();
        });
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            showNext();
        });
    }
    
    // Keyboard handlers inside active lightbox
    document.addEventListener('keydown', (e) => {
        if (!lightboxModal.classList.contains('active')) return;
        
        if (e.key === 'Escape') {
            closeLightbox();
        } else if (e.key === 'ArrowRight') {
            showNext();
        } else if (e.key === 'ArrowLeft') {
            showPrev();
        } else if (e.key === 'Tab') {
            // Keep focus trapped inside lightbox for screenreaders
            const focusableElements = Array.from(lightboxModal.querySelectorAll('button, [tabindex="0"]'));
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];
            
            if (e.shiftKey) { // Shift + Tab
                if (document.activeElement === firstElement) {
                    lastElement.focus();
                    e.preventDefault();
                }
            } else { // Tab
                if (document.activeElement === lastElement) {
                    firstElement.focus();
                    e.preventDefault();
                }
            }
        }
    });
});
