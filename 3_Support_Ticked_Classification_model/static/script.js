document.addEventListener('DOMContentLoaded', () => {
    const classifyBtn = document.getElementById('classify-btn');
    const ticketInput = document.getElementById('ticket-description');
    const resultContainer = document.getElementById('result-container');
    const priorityBadge = document.getElementById('priority-badge');
    const priorityText = document.getElementById('priority-text');
    const priorityDesc = document.getElementById('priority-desc');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');

    const descriptions = {
        'P1': 'Critical / High Urgency. Immediate attention required. This issue severely impacts business operations.',
        'P2': 'Medium Urgency. Important but not critical. Should be addressed as soon as possible.',
        'P3': 'Low Urgency. Standard request or minor issue. Will be resolved in standard queue time.',
        'Unknown': 'Priority could not be determined.'
    };

    classifyBtn.addEventListener('click', async () => {
        const text = ticketInput.value.trim();
        if (!text) {
            alert('Please enter a ticket description.');
            return;
        }

        // UI Loading State
        classifyBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        resultContainer.classList.remove('show');
        
        // Reset classes
        priorityBadge.className = 'priority-badge';
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description: text })
            });

            if (!response.ok) {
                throw new Error('API returned an error');
            }

            const data = await response.json();
            const priority = data.priority || 'Unknown';

            // Update UI
            priorityText.textContent = priority;
            priorityDesc.textContent = descriptions[priority] || descriptions['Unknown'];

            // Style based on priority
            if(priority === 'P1') priorityBadge.classList.add('p1-state');
            else if(priority === 'P2') priorityBadge.classList.add('p2-state');
            else if(priority === 'P3' || priority.startsWith('P')) priorityBadge.classList.add('p3-state');
            else priorityBadge.classList.add('p3-state');
            
            // Show Result
            resultContainer.classList.remove('hidden');
            // Small timeout to allow display:block to apply before animating opacity
            setTimeout(() => {
                resultContainer.classList.add('show');
            }, 50);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while generating prediction. Please try again.');
        } finally {
            classifyBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });
});
