document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target;
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const clusterNum = document.getElementById('cluster-num');
    
    // Add loading state
    form.classList.add('loading');
    resultContainer.classList.add('hidden');
    
    // Gather data
    const formData = new FormData(form);
    const requestData = {};
    
    for (let [key, value] of formData.entries()) {
        requestData[key] = Number(value);
    }
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        // Show result smoothly
        setTimeout(() => {
            clusterNum.textContent = data.cluster;
            const labelText = document.getElementById('cluster-label-text');
            if (labelText && data.label) {
                labelText.textContent = data.label;
            }
            resultContainer.classList.remove('hidden');
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            form.classList.remove('loading');
        }, 500); // Small artificial delay for animation effect
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Check console for details.');
        form.classList.remove('loading');
    }
});
