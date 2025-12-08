document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const predictionForm = document.getElementById('prediction-form');
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const imageInfo = document.getElementById('image-info');
    const fileName = document.getElementById('file-name');
    const removeImageBtn = document.getElementById('remove-image');
    const analyzeButton = document.getElementById('analyze-button');
    const uploadLabel = document.getElementById('upload-label');
    
    // Result elements
    const resultContainer = document.getElementById('result-container');
    const predictionLabel = document.getElementById('prediction-label');
    const predictionBadge = document.getElementById('prediction-badge');
    const resultClass = document.getElementById('result-class');
    const resultProbability = document.getElementById('result-probability');
    const probabilityBar = document.getElementById('probability-bar');
    const recommendation = document.getElementById('recommendation');
    const recommendationText = document.getElementById('recommendation-text');
    
    // Mobile menu elements
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const mobileMenu = document.getElementById('mobile-menu');
    
    let selectedFile = null;
    
    // Mobile menu toggle
    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');
            const icon = mobileMenuBtn.querySelector('i');
            if (mobileMenu.classList.contains('hidden')) {
                icon.className = 'fas fa-bars text-xl';
            } else {
                icon.className = 'fas fa-times text-xl';
            }
        });
        
        // Close menu when clicking links
        const menuLinks = mobileMenu.querySelectorAll('a');
        menuLinks.forEach(link => {
            link.addEventListener('click', function() {
                mobileMenu.classList.add('hidden');
                const icon = mobileMenuBtn.querySelector('i');
                icon.className = 'fas fa-bars text-xl';
            });
        });
    }
    
    // Smooth scrolling
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80;
                window.scrollTo({ top: offsetTop, behavior: 'smooth' });
            }
        });
    });
    
    // Handle image selection
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        handleFileSelect(file);
    });
    
    // Drag and drop
    uploadLabel.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadLabel.classList.add('border-blue-500', 'bg-blue-500/10');
    });
    
    uploadLabel.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadLabel.classList.remove('border-blue-500', 'bg-blue-500/10');
    });
    
    uploadLabel.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadLabel.classList.remove('border-blue-500', 'bg-blue-500/10');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            // Set the file to input element
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            imageUpload.files = dataTransfer.files;
            handleFileSelect(file);
        } else {
            showError('Please upload a valid image file (PNG, JPG, JPEG)');
        }
    });
    
    // Handle file selection
    function handleFileSelect(file) {
        if (!file) return;
        
        if (!file.type.startsWith('image/')) {
            showError('Please upload a valid image file');
            return;
        }
        
        if (file.size > 10 * 1024 * 1024) {
            showError('File size must be less than 10MB');
            return;
        }
        
        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            uploadPlaceholder.classList.add('hidden');
        };
        reader.readAsDataURL(file);
        
        // Show file info
        fileName.textContent = file.name;
        imageInfo.classList.remove('hidden');
        
        // Enable button
        analyzeButton.disabled = false;
        
        // Hide results
        resultContainer.classList.add('hidden');
    }
    
    // Remove image
    removeImageBtn.addEventListener('click', function() {
        selectedFile = null;
        imageUpload.value = '';
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
        imageInfo.classList.add('hidden');
        analyzeButton.disabled = true;
        resultContainer.classList.add('hidden');
    });
    
    // Form submission
    predictionForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        if (!selectedFile) {
            showError('Please select an image first');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Loading state
        const originalHTML = analyzeButton.innerHTML;
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = `
            <svg class="animate-spin -ml-1 mr-2 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Analyzing...
        `;
        
        // Send request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Something went wrong');
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                displayResult(data);
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        })
        .catch(error => {
            showError(error.message);
        })
        .finally(() => {
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = originalHTML;
        });
    });
    
    // Display results
    function displayResult(data) {
        const isFresh = data.status.toLowerCase() === 'segar' || data.class.toLowerCase().includes('fresh');
        
        // Set label
        predictionLabel.textContent = isFresh ? 'FRESH' : 'ROTTEN';
        predictionLabel.className = isFresh ? 'text-green-400' : 'text-red-400';
        
        // Set badge
        if (isFresh) {
            predictionBadge.textContent = '✓ Safe to Eat';
            predictionBadge.className = 'px-4 py-2 rounded-full text-sm font-bold bg-green-500/20 text-green-400 border border-green-500/30';
            probabilityBar.className = 'h-full rounded-full transition-all duration-1000 bg-gradient-to-r from-green-400 to-green-600';
        } else {
            predictionBadge.textContent = '✗ Not Safe';
            predictionBadge.className = 'px-4 py-2 rounded-full text-sm font-bold bg-red-500/20 text-red-400 border border-red-500/30';
            probabilityBar.className = 'h-full rounded-full transition-all duration-1000 bg-gradient-to-r from-red-400 to-red-600';
        }
        
        // Format class name
        let foodType = data.class.replace('fresh', '').replace('rotten', '');
        foodType = foodType.charAt(0).toUpperCase() + foodType.slice(1);
        resultClass.textContent = foodType;
        
        // Set confidence
        const confidence = data.confidence;
        resultProbability.textContent = confidence + '%';
        
        // Animate progress bar
        setTimeout(() => {
            probabilityBar.style.width = confidence + '%';
        }, 100);
        
        // Display AI Explanation
        if (data.ai_explanation) {
            const explanation = data.ai_explanation;
            const explanationColor = isFresh ? 'green' : 'red';
            
            // Build AI explanation HTML
            let explanationHTML = `
                <div class="mt-6 p-6 rounded-xl bg-${explanationColor}-500/10 border border-${explanationColor}-500/30">
                    <div class="flex items-start space-x-3 mb-4">
                        <i class="fas fa-robot text-2xl text-${explanationColor}-400 mt-1"></i>
                        <div class="flex-1">
                            <h4 class="font-bold text-lg text-${explanationColor}-300 mb-2">${explanation.title}</h4>
                            <p class="text-gray-300 text-sm leading-relaxed">${explanation.description}</p>
                        </div>
                    </div>
                    
                    <div class="mt-4 p-4 bg-dark/30 rounded-lg">
                        <h5 class="font-semibold text-sm text-${explanationColor}-300 mb-2 flex items-center">
                            <i class="fas fa-microscope mr-2"></i>AI Analysis:
                        </h5>
                        <ul class="space-y-1 text-sm text-gray-300">
                            ${explanation.analysis.map(point => `<li class="flex items-start"><span class="mr-2">•</span><span>${point}</span></li>`).join('')}
                        </ul>
                    </div>
                    
                    <div class="mt-4 p-4 bg-dark/30 rounded-lg">
                        <h5 class="font-semibold text-sm text-${explanationColor}-300 mb-2 flex items-center">
                            <i class="fas fa-lightbulb mr-2"></i>Recommendation:
                        </h5>
                        <p class="text-sm text-gray-300 leading-relaxed">${explanation.recommendation}</p>
                    </div>
            `;
            
            // Add storage tips or health risk based on status
            if (explanation.storage_tips) {
                explanationHTML += `
                    <div class="mt-4 p-4 bg-dark/30 rounded-lg">
                        <h5 class="font-semibold text-sm text-${explanationColor}-300 mb-2 flex items-center">
                            <i class="fas fa-box mr-2"></i>Storage Tips:
                        </h5>
                        <p class="text-sm text-gray-300 leading-relaxed">${explanation.storage_tips}</p>
                    </div>
                `;
            }
            
            if (explanation.health_benefit) {
                explanationHTML += `
                    <div class="mt-4 p-4 bg-dark/30 rounded-lg">
                        <h5 class="font-semibold text-sm text-${explanationColor}-300 mb-2 flex items-center">
                            <i class="fas fa-heartbeat mr-2"></i>Health Benefits:
                        </h5>
                        <p class="text-sm text-gray-300 leading-relaxed">${explanation.health_benefit}</p>
                    </div>
                `;
            }
            
            if (explanation.health_risk) {
                explanationHTML += `
                    <div class="mt-4 p-4 bg-red-900/30 rounded-lg border border-red-500/30">
                        <h5 class="font-semibold text-sm text-red-300 mb-2 flex items-center">
                            <i class="fas fa-exclamation-triangle mr-2"></i>Health Risk Warning:
                        </h5>
                        <p class="text-sm text-red-200 leading-relaxed font-medium">${explanation.health_risk}</p>
                    </div>
                `;
            }
            
            if (explanation.disposal) {
                explanationHTML += `
                    <div class="mt-4 p-4 bg-dark/30 rounded-lg">
                        <h5 class="font-semibold text-sm text-${explanationColor}-300 mb-2 flex items-center">
                            <i class="fas fa-trash-alt mr-2"></i>Disposal Instructions:
                        </h5>
                        <p class="text-sm text-gray-300 leading-relaxed">${explanation.disposal}</p>
                    </div>
                `;
            }
            
            explanationHTML += `</div>`;
            
            // Update recommendation section with AI explanation
            recommendation.innerHTML = explanationHTML;
            recommendation.className = `mt-6`;
        } else {
            // Fallback to simple recommendation if no AI explanation
            recommendation.className = `mt-6 p-4 rounded-xl bg-${isFresh ? 'green' : 'red'}-500/10 border border-${isFresh ? 'green' : 'red'}-500/30 text-${isFresh ? 'green' : 'red'}-100`;
            recommendationText.textContent = isFresh 
                ? 'This food appears to be fresh and safe to consume. Store it properly to maintain its quality.'
                : 'This food appears to be rotten. We recommend not consuming it to avoid potential health risks.';
        }
        
        // Show results with animation
        resultContainer.classList.remove('hidden');
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // Show error
    function showError(message) {
        const existingAlerts = document.querySelectorAll('.error-alert');
        existingAlerts.forEach(alert => alert.remove());
        
        const alertElement = document.createElement('div');
        alertElement.className = 'error-alert fixed top-24 right-4 z-50 max-w-md px-6 py-4 bg-red-500/90 backdrop-blur-md text-white rounded-xl shadow-2xl border border-red-400/30';
        alertElement.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0">
                    <i class="fas fa-exclamation-circle text-2xl"></i>
                </div>
                <div class="ml-3 flex-1">
                    <p class="font-semibold text-sm">${message}</p>
                </div>
                <button class="close-alert ml-4 text-white/80 hover:text-white">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(alertElement);
        
        alertElement.querySelector('.close-alert').addEventListener('click', function() {
            alertElement.remove();
        });
        
        setTimeout(() => {
            if (document.body.contains(alertElement)) {
                alertElement.style.opacity = '0';
                alertElement.style.transform = 'translateX(100%)';
                setTimeout(() => alertElement.remove(), 300);
            }
        }, 5000);
    }
});
