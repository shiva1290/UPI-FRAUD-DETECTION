// Dashboard JavaScript - Enhanced with Advanced Features

// Load data on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadModelPerformance();
    loadHourlyFraud();
    loadFeatureImportance();
    loadRecentTransactions();
    loadLLMSamples();
});

// Load overall statistics
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        // Update header stats
        document.getElementById('totalTransactions').textContent = data.total_transactions.toLocaleString();
        document.getElementById('fraudCount').textContent = data.fraud_count.toLocaleString();
        
        // Update stat cards
        document.getElementById('totalTxns').textContent = data.total_transactions.toLocaleString();
        document.getElementById('fraudTxns').textContent = data.fraud_count.toLocaleString();
        document.getElementById('legitTxns').textContent = data.legitimate_count.toLocaleString();
        document.getElementById('fraudPercent').textContent = data.fraud_percentage.toFixed(2) + '%';
        document.getElementById('avgAmount').textContent = '₹' + Math.round(data.avg_transaction_amount).toLocaleString();
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Load model performance
async function loadModelPerformance() {
    try {
        const response = await fetch('/api/model_performance');
        const models = await response.json();
        
        // Create table
        const tbody = document.getElementById('modelTableBody');
        tbody.innerHTML = '';
        
        let bestF1 = 0;
        let bestModel = null;
        
        models.forEach(model => {
            const row = document.createElement('tr');
            if (model.f1_score > bestF1) {
                bestF1 = model.f1_score;
                bestModel = row;
            }
            
            const rawName = (model.model || '').trim();
            const modelName = rawName
                ? rawName.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ')
                : 'Unknown';
            
            row.innerHTML = `
                <td><span class="model-name">${modelName}</span></td>
                <td>${(model.accuracy * 100).toFixed(2)}%</td>
                <td>${(model.precision * 100).toFixed(2)}%</td>
                <td>${(model.recall * 100).toFixed(2)}%</td>
                <td>${(model.f1_score * 100).toFixed(2)}%</td>
                <td>${model.roc_auc ? (model.roc_auc * 100).toFixed(2) + '%' : 'N/A'}</td>
            `;
            tbody.appendChild(row);
        });
        
        if (bestModel) {
            bestModel.classList.add('best-model');
        }
        
        // Create chart
        const ctx = document.getElementById('modelPerformanceChart').getContext('2d');
        const formatModelLabel = (name) => (name || '').trim()
            ? (name || '').trim().split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ')
            : 'Unknown';
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: models.map(m => formatModelLabel(m.model)),
                datasets: [
                    {
                        label: 'Accuracy',
                        data: models.map(m => m.accuracy * 100),
                        backgroundColor: 'rgba(99, 102, 241, 0.8)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'Precision',
                        data: models.map(m => m.precision * 100),
                        backgroundColor: 'rgba(16, 185, 129, 0.8)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'Recall',
                        data: models.map(m => m.recall * 100),
                        backgroundColor: 'rgba(245, 158, 11, 0.8)',
                        borderColor: 'rgba(245, 158, 11, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'F1-Score',
                        data: models.map(m => m.f1_score * 100),
                        backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Model Performance Metrics (%)',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error loading model performance:', error);
    }
}

// Load hourly fraud distribution
async function loadHourlyFraud() {
    try {
        const response = await fetch('/api/hourly_fraud');
        const data = await response.json();
        
        const ctx = document.getElementById('hourlyChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.hours,
                datasets: [{
                    label: 'Fraud Rate (%)',
                    data: data.fraud_rate,
                    borderColor: 'rgba(239, 68, 68, 1)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Fraud Rate (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Hour of Day'
                        }
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error loading hourly fraud:', error);
    }
}

// Load feature importance
async function loadFeatureImportance() {
    try {
        const response = await fetch('/api/feature_importance');
        const data = await response.json();
        
        const ctx = document.getElementById('featureChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.features,
                datasets: [{
                    label: 'Importance',
                    data: data.importance,
                    backgroundColor: 'rgba(99, 102, 241, 0.8)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error loading feature importance:', error);
    }
}

// Load recent transactions
async function loadRecentTransactions() {
    try {
        const response = await fetch('/api/recent_transactions');
        const transactions = await response.json();
        
        const tbody = document.getElementById('transactionTable');
        tbody.innerHTML = '';
        
        transactions.slice(0, 20).forEach(txn => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${txn.transaction_id}</td>
                <td>₹${Math.round(txn.amount).toLocaleString()}</td>
                <td>${txn.hour}:00</td>
                <td>${txn.transaction_velocity}</td>
                <td>${txn.device_change ? '⚠️ Changed' : '✓ Same'}</td>
                <td>${txn.location_change_km.toFixed(1)} km</td>
                <td>
                    <span class="status-badge ${txn.is_fraud ? 'fraud' : 'legitimate'}">
                        ${txn.is_fraud ? 'FRAUD' : 'LEGITIMATE'}
                    </span>
                </td>
            `;
            tbody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Error loading transactions:', error);
    }
}

// Load LLM samples
async function loadLLMSamples() {
    try {
        const response = await fetch('/api/llm_samples');
        const samples = await response.json();
        
        const container = document.getElementById('llmSamples');
        container.innerHTML = '';
        
        if (samples.error || samples.length === 0) {
            container.innerHTML = '<div class="loading">⚠️ LLM data not available. Run training with --with-llm flag to generate LLM predictions.</div>';
            return;
        }
        
        samples.forEach(sample => {
            const card = document.createElement('div');
            const isFraud = Number(sample.llm_prediction) === 1;
            card.className = `llm-card ${isFraud ? 'fraud' : ''}`;
            // risk_factors from CSV may be Python-style string "['a','b']"; support both that and llm_risk_factors
            const rawFactors = sample.llm_risk_factors ?? sample.risk_factors ?? '[]';
            let factors = [];
            try {
                factors = typeof rawFactors === 'string'
                    ? (rawFactors.trim().startsWith('[') ? JSON.parse(rawFactors.replace(/'/g, '"')) : [])
                    : (Array.isArray(rawFactors) ? rawFactors : []);
            } catch (_) {
                factors = [];
            }
            const confidence = sample.llm_confidence != null ? Number(sample.llm_confidence).toFixed(0) : '0';
            card.innerHTML = `
                <div class="llm-header">
                    <span class="llm-prediction ${isFraud ? 'fraud' : 'legitimate'}">
                        ${isFraud ? '⚠️ FRAUD' : '✅ LEGITIMATE'}
                    </span>
                    <span class="llm-confidence">
                        Confidence: ${confidence}%
                    </span>
                </div>
                <div class="llm-reasoning">${(sample.llm_reasoning ?? '').toString()}</div>
                <div class="llm-factors">
                    ${factors.map(f => `<span class="factor-tag">• ${f}</span>`).join('')}
                </div>
            `;
            container.appendChild(card);
        });
        
    } catch (error) {
        console.error('Error loading LLM samples:', error);
        document.getElementById('llmSamples').innerHTML = '<div class="loading">⚠️ LLM data not available. Run training with --with-llm flag.</div>';
    }
}

// Test transaction with ML model
async function testTransaction() {
    const transaction = buildTransactionData();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(transaction)
        });
        
        const result = await response.json();
        
        // Display result
        displayPrediction(result, 'ML Model');
        
    } catch (error) {
        console.error('Error testing transaction:', error);
        alert('Error analyzing transaction. Please try again.');
    }
}

// Test transaction with LLM
async function testWithLLM() {
    const transaction = buildTransactionData();
    
    if (!confirm('This will use your Groq API credits. Continue?')) {
        return;
    }
    
    try {
        const response = await fetch('/api/predict_llm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(transaction)
        });
        
        const result = await response.json();
        
        if (result.error) {
            alert('LLM Error: ' + result.error);
            return;
        }
        
        // Display LLM result with reasoning
        displayLLMPrediction(result);
        
    } catch (error) {
        console.error('Error testing with LLM:', error);
        alert('Error analyzing with LLM. Make sure API key is configured.');
    }
}

// Build transaction data from form inputs
function buildTransactionData() {
    const dayOfWeek = parseInt(document.getElementById('inputDay').value);
    const hour = parseInt(document.getElementById('inputHour').value);
    
    return {
        amount: parseFloat(document.getElementById('inputAmount').value),
        hour: hour,
        day_of_week: dayOfWeek,
        transaction_velocity: parseInt(document.getElementById('inputVelocity').value),
        failed_attempts: parseInt(document.getElementById('inputFailed').value),
        device_change: parseInt(document.getElementById('inputDevice').value),
        location_change_km: parseFloat(document.getElementById('inputLocation').value),
        location_lat: parseFloat(document.getElementById('inputLat').value),
        location_lon: parseFloat(document.getElementById('inputLon').value),
        is_weekend: dayOfWeek >= 5 ? 1 : 0,
        is_night: (hour >= 22 || hour <= 5) ? 1 : 0,
        
        // Advanced features
        amount_deviation_pct: parseFloat(document.getElementById('inputDeviation').value),
        reversed_attempts: parseInt(document.getElementById('inputReversed').value),
        is_new_beneficiary: parseInt(document.getElementById('inputNewBenef').value),
        beneficiary_fan_in: parseInt(document.getElementById('inputFanIn').value),
        beneficiary_account_age_days: parseInt(document.getElementById('inputAccAge').value),
        approval_delay_sec: parseFloat(document.getElementById('inputApproval').value),
        is_multiple_device_accounts: parseInt(document.getElementById('inputMultiAcc').value),
        network_switch: parseInt(document.getElementById('inputNetSwitch').value),
        recent_unknown_call: parseInt(document.getElementById('inputUnknownCall').value),
        has_suspicious_keywords: parseInt(document.getElementById('inputKeywords').value),
        merchant_category_mismatch: parseInt(document.getElementById('inputMerchantMismatch').value),
        fraud_network_proximity: parseInt(document.getElementById('inputFraudNetwork').value)
    };
}

// Display ML prediction result
function displayPrediction(result, source) {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.innerHTML = `
        <div class="prediction-card">
            <div style="font-size: 0.875rem; color: var(--text-light); margin-bottom: 1rem;">
                ${source} Prediction
            </div>
            <div class="prediction-badge ${result.prediction.toLowerCase()}">
                ${result.prediction === 'FRAUD' ? '⚠️' : '✅'} ${result.prediction}
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${result.probability * 100}%">
                    ${(result.probability * 100).toFixed(1)}%
                </div>
            </div>
            <div class="risk-level">
                Risk Level: <strong>${result.risk_level}</strong>
            </div>
        </div>
    `;
}

// Display LLM prediction with reasoning
function displayLLMPrediction(result) {
    const resultDiv = document.getElementById('predictionResult');
    const riskFactors = result.risk_factors || [];
    
    resultDiv.innerHTML = `
        <div class="prediction-card">
            <div style="font-size: 0.875rem; color: var(--text-light); margin-bottom: 1rem;">
                🤖 LLM Analysis (Groq API)
            </div>
            <div class="prediction-badge ${result.prediction.toLowerCase()}">
                ${result.prediction === 'FRAUD' ? '⚠️' : '✅'} ${result.prediction}
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${result.confidence * 100}%">
                    ${(result.confidence * 100).toFixed(0)}% Confidence
                </div>
            </div>
            <div style="margin-top: 1.5rem; padding: 1rem; background: var(--light); border-radius: 0.5rem;">
                <strong>💡 Reasoning:</strong>
                <p style="margin-top: 0.5rem; line-height: 1.6;">${result.reasoning}</p>
            </div>
            ${riskFactors.length > 0 ? `
                <div style="margin-top: 1rem;">
                    <strong>🚨 Risk Factors:</strong>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
                        ${riskFactors.map(f => `<span class="factor-tag">• ${f}</span>`).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}
