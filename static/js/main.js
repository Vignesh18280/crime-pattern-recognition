document.addEventListener('DOMContentLoaded', () => {
    const categorySelects = document.querySelectorAll('.category-select');
    const compareBtn = document.getElementById('compareBtn');
    const resultsSection = document.getElementById('results');
    const loader = document.getElementById('loader');
    const verdictEl = document.getElementById('verdict');
    const similarityScoreEl = document.getElementById('similarityScore');
    const shapChartCanvas = document.getElementById('shapChart');
    let shapChart = null;

    const generateBtns = document.querySelectorAll('.generate-btn');

    const dropZones = {
        'drop-zone-1': { fileInput: document.getElementById('file-input-1'), statusElement: document.getElementById('upload-status-1'), uploadedFilename: null, categorySelect: document.getElementById('category1') },
        'drop-zone-2': { fileInput: document.getElementById('file-input-2'), statusElement: document.getElementById('upload-status-2'), uploadedFilename: null, categorySelect: document.getElementById('category2') }
    };

    // --- Helper for Drag & Drop ---
    function setupDropZone(dropZoneId) {
        const dropZone = document.getElementById(dropZoneId);
        const fileInput = dropZones[dropZoneId].fileInput;
        // const statusElement = dropZones[dropZoneId].statusElement; // No longer needed here
        // const categorySelect = dropZones[dropZoneId].categorySelect; // No longer needed here

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('hover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('hover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('hover');
            dropZone.classList.remove('error'); // Clear error state on new drop
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0], dropZoneId);
            }
        });

        fileInput.addEventListener('change', (e) => {
            dropZone.classList.remove('error'); // Clear error state on new selection
            const files = e.target.files;
            if (files.length > 0) {
                handleFileUpload(files[0], dropZoneId);
            }
        });

        dropZone.addEventListener('click', () => {
            fileInput.click(); // Trigger hidden file input click
        });

        // Set initial text
        dropZone.textContent = (dropZoneId === 'drop-zone-1') ? 'Drag & Drop File 1 (5 events)' : 'Drag & Drop File 2 (5 events)';
    }

    async function handleFileUpload(file, dropZoneId) {
        const { statusElement, categorySelect } = dropZones[dropZoneId];
        const dropZone = document.getElementById(dropZoneId);

        statusElement.textContent = `Uploading ${file.name}...`;
        dropZone.classList.remove('active');
        dropZone.classList.remove('error');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'File upload failed.');
            }

            const data = await response.json();
            dropZones[dropZoneId].uploadedFilename = data.filename;
            statusElement.textContent = `Uploaded: ${data.filename}`;
            dropZone.classList.add('active');
            dropZone.textContent = data.filename; // Display filename on dropzone
            checkCompareButtonState();

            // Clear category selection for this pattern
            categorySelect.value = '';

        } catch (error) {
            console.error('Upload error:', error);
            statusElement.textContent = `Error: ${error.message}`;
            dropZone.classList.add('error');
            dropZones[dropZoneId].uploadedFilename = null; // Clear filename on error
            checkCompareButtonState();
            dropZone.textContent = (dropZoneId === 'drop-zone-1') ? 'Drag & Drop File 1 (5 events)' : 'Drag & Drop File 2 (5 events)';
        }
    }

    // --- Generate Dummy Pattern Button Event ---
    generateBtns.forEach(button => {
        button.addEventListener('click', async () => {
            const patternNum = button.dataset.pattern;
            const dropZoneId = `drop-zone-${patternNum}`;
            const { statusElement, categorySelect } = dropZones[dropZoneId];
            const dropZone = document.getElementById(dropZoneId);

            statusElement.textContent = 'Generating dummy pattern...';
            dropZone.classList.remove('active');
            dropZone.classList.remove('error');

            try {
                // Optionally send a base_category to make dummy data statistically similar
                // For now, we'll keep it simple and just generate a random one
                const response = await fetch('/api/generate_dummy_pattern', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ base_category: categorySelect.value || null }) // Use selected category as base if available
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Dummy pattern generation failed.');
                }

                const data = await response.json();
                const filename = data.filename;
                const downloadUrl = data.download_url;

                // Initiate download
                const a = document.createElement('a');
                a.href = downloadUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);

                dropZones[dropZoneId].uploadedFilename = filename;
                statusElement.textContent = `Generated & Downloaded: ${filename}. Please drag & drop it here.`;
                dropZone.classList.add('active'); // Visually indicate a file is associated
                dropZone.textContent = filename; // Show filename on dropzone
                checkCompareButtonState();

                // Clear category selection
                categorySelect.value = '';

            } catch (error) {
                console.error('Generate dummy pattern error:', error);
                statusElement.textContent = `Error: ${error.message}`;
                dropZone.classList.add('error');
                dropZones[dropZoneId].uploadedFilename = null;
                checkCompareButtonState();
                dropZone.textContent = (dropZoneId === 'drop-zone-1') ? 'Drag & Drop File 1 (5 events)' : 'Drag & Drop File 2 (5 events)';
            }
        });
    });


    // --- Initialize Drop Zones ---
    setupDropZone('drop-zone-1');
    setupDropZone('drop-zone-2');

    // --- Fetch Attack Categories ---
    fetch('/api/attack_categories')
        .then(response => response.json())
        .then(categories => {
            if (categories && categories.length > 0) {
                categorySelects.forEach(select => {
                    // Add a default "Select Category" option
                    const defaultOption = document.createElement('option');
                    defaultOption.value = '';
                    defaultOption.textContent = 'Select Category';
                    select.appendChild(defaultOption);

                    categories.forEach(category => {
                        const option = document.createElement('option');
                        option.value = category;
                        option.textContent = category;
                        select.appendChild(option);
                    });
                     // Add event listener to clear uploaded file if category selected
                    select.addEventListener('change', (e) => {
                        const selectedCategory = e.target.value;
                        const dropZoneKey = (e.target.id === 'category1') ? 'drop-zone-1' : 'drop-zone-2';
                        if (selectedCategory !== '' && dropZones[dropZoneKey].uploadedFilename) {
                            dropZones[dropZoneKey].uploadedFilename = null;
                            dropZones[dropZoneKey].statusElement.textContent = '';
                            document.getElementById(dropZoneKey).classList.remove('active');
                             document.getElementById(dropZoneKey).textContent = (dropZoneKey === 'drop-zone-1') ? 'Drag & Drop File 1 (5 events)' : 'Drag & Drop File 2 (5 events)';
                        }
                        checkCompareButtonState();
                    });
                });
            } else {
                console.warn("No categories loaded from API. Only file upload will be possible.");
                // Disable category selects if no categories
                categorySelects.forEach(select => select.disabled = true);
            }
            checkCompareButtonState(); // Initial check
        })
        .catch(error => {
            console.error('Error fetching categories:', error);
            categorySelects.forEach(select => select.disabled = true);
            checkCompareButtonState(); // Initial check
        });

    // --- Enable/Disable Compare Button ---
    function checkCompareButtonState() {
        const category1 = document.getElementById('category1').value;
        const category2 = document.getElementById('category2').value;
        const file1 = dropZones['drop-zone-1'].uploadedFilename;
        const file2 = dropZones['drop-zone-2'].uploadedFilename;

        const isPattern1Ready = (category1 !== '' || file1 !== null);
        const isPattern2Ready = (category2 !== '' || file2 !== null);

        compareBtn.disabled = !(isPattern1Ready && isPattern2Ready);
    }
    
    // Listen for changes in category selects to update button state
    categorySelects.forEach(select => select.addEventListener('change', checkCompareButtonState));


    // --- Compare Button Event ---
    compareBtn.addEventListener('click', () => {
        const category1 = document.getElementById('category1').value;
        const category2 = document.getElementById('category2').value;
        const uploadedFile1 = dropZones['drop-zone-1'].uploadedFilename;
        const uploadedFile2 = dropZones['drop-zone-2'].uploadedFilename;

        let payload = {};
        if (uploadedFile1) {
            payload.uploadedFile1 = uploadedFile1;
        } else if (category1) {
            payload.category1 = category1;
        } else {
            alert('Please select a category or upload a file for Crime Pattern A.');
            return;
        }

        if (uploadedFile2) {
            payload.uploadedFile2 = uploadedFile2;
        } else if (category2) {
            payload.category2 = category2;
        } else {
            alert('Please select a category or upload a file for Crime Pattern B.');
            return;
        }
        
        loader.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Network response was not ok'); });
            }
            return response.json();
        })
        .then(data => {
            loader.classList.add('hidden');
            displayResults(data);
        })
        .catch(error => {
            loader.classList.add('hidden');
            console.error('Error during prediction:', error);
            alert(`An error occurred while analyzing the patterns: ${error.message || error}. Please check the console.`);
        });
    });

    // --- Display Results ---
    function displayResults(data) {
        resultsSection.classList.remove('hidden');

        // Display Verdict
        verdictEl.textContent = data.verdict;
        verdictEl.className = 'verdict-card'; // Reset classes
        if (data.verdict === 'SAME MO') {
            verdictEl.classList.add('same-mo');
        } else {
            verdictEl.classList.add('different-mo');
        }

        // Display Similarity Score
        similarityScoreEl.textContent = parseFloat(data.similarityScore).toFixed(4);

        // Display SHAP Chart
        if (data.explanation && data.explanation.length > 0) {
            const labels = data.explanation.map(item => item.feature);
            const importances = data.explanation.map(item => item.importance);
            const backgroundColors = importances.map(val => val > 0 ? 'rgba(42, 157, 143, 0.6)' : 'rgba(231, 111, 81, 0.6)');
            const borderColors = importances.map(val => val > 0 ? 'rgba(42, 157, 143, 1)' : 'rgba(231, 111, 81, 1)');

            if (shapChart) {
                shapChart.destroy();
            }

            shapChart = new Chart(shapChartCanvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Feature Importance',
                        data: importances,
                        backgroundColor: backgroundColors,
                        borderColor: borderColors,
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#f5f6f7'
                            },
                            title: {
                                display: true,
                                text: 'Impact on Similarity (Positive = Same MO, Negative = Different MO)',
                                color: '#f5f6f7'
                            }
                        },
                        y: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#f5f6f7'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Top Features Influencing Similarity',
                            color: '#f5f6f7'
                        }
                    }
                }
            });
        }
    }
});
