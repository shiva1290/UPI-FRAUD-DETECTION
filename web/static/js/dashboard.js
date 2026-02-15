// Dashboard JavaScript - Enhanced with Advanced Features

// Load data on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadModelPerformance();
    loadHourlyFraud();
    loadFeatureImportance();
    loadRecentPredictions();
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

// Load recent predictions (from tester) - with optional LLM analyze button
async function loadRecentPredictions() {
    try {
        const response = await fetch('/api/recent_predictions');
        const predictions = await response.json();
        
        const tbody = document.getElementById('recentPredictionsTable');
        tbody.innerHTML = '';
        
        if (!predictions || predictions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="loading">No predictions yet. Use the tester above.</td></tr>';
            return;
        }
        
        predictions.slice().reverse().slice(0, 20).forEach(pred => {
            const row = document.createElement('tr');
            const td = pred.transaction_data || {};
            const shortId = (pred.id || '').substring(0, 8);
            const riskLevel = pred.risk_level || 'Unknown';
            const showLLMButton = pred.suggest_llm && !pred.llm_analyzed;
            const decisionMsg = getDecisionMessage(pred.action);
            row.innerHTML = `
                <td><code>${shortId}</code></td>
                <td>₹${Math.round(td.amount || 0).toLocaleString()}</td>
                <td>${td.hour != null ? td.hour + ':00' : '-'}</td>
                <td><span class="risk-score-inline">${(pred.risk_score != null ? pred.risk_score.toFixed(1) : '-')}</span></td>
                <td><span class="status-badge risk-${riskLevel.toLowerCase()}">${riskLevel}</span></td>
                <td><span class="decision-badge action-${(pred.action || 'allow').toLowerCase()}">${decisionMsg}</span></td>
                <td>
                    ${pred.llm_analyzed ? '<span class="llm-done">✓ LLM done</span>' : (showLLMButton ? `<button class="btn-llm-small" onclick="analyzeWithLLM('${pred.id}')">Analyze with LLM</button>` : '-')}
                </td>
            `;
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading recent predictions:', error);
        document.getElementById('recentPredictionsTable').innerHTML = '<tr><td colspan="7" class="loading">Error loading predictions.</td></tr>';
    }
}

// Load recent transactions (dataset sample)
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
        
        // Display result (includes "Analyze with LLM?" for medium/high)
        displayPrediction(result, 'ML Model');
        loadRecentPredictions();
        
    } catch (error) {
        console.error('Error testing transaction:', error);
        alert('Error analyzing transaction. Please try again.');
    }
}

// Analyze existing prediction with LLM (for medium/high risk)
async function analyzeWithLLM(predictionId) {
    if (!confirm('This will use your Groq API credits. Continue?')) {
        return;
    }
    try {
        const response = await fetch('/api/predict_llm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prediction_id: predictionId })
        });
        const result = await response.json();
        if (result.error) {
            alert('LLM Error: ' + result.error);
            return;
        }
        displayPrediction(result, result.llm_analyzed ? 'ML + LLM' : 'LLM');
        loadRecentPredictions();
    } catch (error) {
        console.error('Error analyzing with LLM:', error);
        alert('Error analyzing with LLM. Please try again.');
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

// Map backend action to user-facing decision message (SRP: single source of truth)
function getDecisionMessage(action) {
    const map = { allow: 'Allowed', review: 'Suspicious', block: 'Flagged' };
    return map[(action || 'allow').toLowerCase()] || action;
}

// Risk score color for gauge/bar (Low=green, Medium=amber, High=red)
function getRiskColor(riskLevel) {
    const map = { low: '#10b981', medium: '#f59e0b', high: '#ef4444' };
    return map[(riskLevel || '').toLowerCase()] || '#8b5cf6';
}

// Build SVG semi-circular gauge for risk 0–100
function buildRiskGaugeSvg(riskScore, riskLevel) {
    const pct = Math.min(Math.max(riskScore, 0), 100) / 100;
    const color = getRiskColor(riskLevel);
    const pathLen = Math.PI * 50;
    const dashLen = pct * pathLen;
    return `
        <svg class="risk-gauge-svg" viewBox="0 0 120 80" preserveAspectRatio="xMidYMid meet">
            <defs>
                <linearGradient id="gaugeBg" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#10b981"/>
                    <stop offset="30%" stop-color="#10b981"/>
                    <stop offset="30%" stop-color="#f59e0b"/>
                    <stop offset="70%" stop-color="#f59e0b"/>
                    <stop offset="70%" stop-color="#ef4444"/>
                    <stop offset="100%" stop-color="#ef4444"/>
                </linearGradient>
            </defs>
            <path class="gauge-bg" d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="url(#gaugeBg)" stroke-width="10" stroke-linecap="round"/>
            <path class="gauge-fill" d="M 10 70 A 50 50 0 0 1 110 70" fill="none" stroke="${color}" stroke-width="10" stroke-linecap="round" stroke-dasharray="${dashLen} ${pathLen}"/>
            <text x="60" y="55" class="gauge-text" text-anchor="middle">${(riskScore || 0).toFixed(0)}</text>
        </svg>
    `;
}

// Display prediction result (risk score + risk level + decision + explanation)
function displayPrediction(result, source) {
    const resultDiv = document.getElementById('predictionResult');
    const riskScore = result.risk_score != null ? result.risk_score : (result.probability != null ? result.probability * 100 : 0);
    const riskLevel = result.risk_level || 'Unknown';
    const action = result.action || 'allow';
    const actionBadge = action === 'block' ? 'block' : (action === 'review' ? 'review' : 'allow');
    const decisionMessage = getDecisionMessage(action);
    const explanation = result.explanation || null;
    const showLLMButton = result.suggest_llm && !result.llm_analyzed && result.id;
    const isRisky = (riskLevel === 'Medium' || riskLevel === 'High');
    const contributingFeatures = result.contributing_features || [];
    const riskFactors = result.risk_factors || [];
    const riskColor = getRiskColor(riskLevel);
    const mlLatency = result.ml_latency_ms;
    const llmLatency = result.llm_latency_ms;
    const latencyHtml = (mlLatency != null || llmLatency != null) ? `
            <div class="latency-section">
                <h4 class="latency-title">Latency (ML vs LLM)</h4>
                <div class="latency-grid">
                    ${mlLatency != null ? `<div class="latency-item"><span class="latency-label">ML:</span> <strong>${mlLatency} ms</strong></div>` : ''}
                    ${llmLatency != null ? `<div class="latency-item"><span class="latency-label">LLM:</span> <strong>${llmLatency} ms</strong></div>` : ''}
                </div>
                ${(mlLatency != null && llmLatency != null) ? `<div class="latency-comparison">LLM adds ~${Math.round(llmLatency)} ms over ML-only (${mlLatency} ms)</div>` : ''}
            </div>
    ` : '';

    resultDiv.innerHTML = `
        <div class="prediction-card decision-support">
            <div class="prediction-source">${source} – Risk Assessment</div>
            <div class="risk-gauge-stack">
                <div class="risk-gauge-wrapper">${buildRiskGaugeSvg(riskScore, riskLevel)}</div>
                <div class="risk-level-below">
                    <span class="prediction-badge risk-level-${riskLevel.toLowerCase()}">${riskLevel}</span>
                </div>
            </div>
            <div class="decision-message action-${actionBadge}">
                <span class="decision-label">Decision:</span>
                <strong>${decisionMessage}</strong>
            </div>
            ${latencyHtml}
            ${showLLMButton ? `
                <div class="llm-cta">
                    <button class="btn-llm-cta" onclick="analyzeWithLLM('${result.id}')">
                        🤖 Check with LLM?
                    </button>
                    <span class="llm-cta-hint">Optional: Get human-readable reasoning for this ${riskLevel} risk transaction</span>
                </div>
            ` : ''}
            ${(explanation || contributingFeatures.length > 0 || riskFactors.length > 0) ? `
                <div class="explanation-section ${isRisky ? 'risky' : ''}">
                    <h4 class="explanation-section-title">Decision Rationale</h4>
                    ${explanation ? `
                        <div class="explanation-box">
                            <p>${String(explanation).replace(/\n/g, '<br>')}</p>
                        </div>
                    ` : ''}
                    ${contributingFeatures.length > 0 ? `
                        <div class="contributing-features">
                            <strong>Contributing Factors:</strong>
                            <div class="factor-tags">
                                ${contributingFeatures.map(f => `<span class="factor-tag">• ${f}</span>`).join('')}
                            </div>
                        </div>
                    ` : ''}
                    ${riskFactors.length > 0 ? `
                        <div class="risk-factors">
                            <strong>Risk Factors:</strong>
                            <div class="factor-tags">
                                ${riskFactors.map(f => `<span class="factor-tag">• ${f}</span>`).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            ` : ''}
        </div>
    `;
}

// Display LLM prediction (uses same risk format: risk_score, risk_level, explanation)
function displayLLMPrediction(result) {
    displayPrediction(result, '🤖 LLM (Groq API)');
}
