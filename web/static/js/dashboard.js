// Dashboard JavaScript - Enhanced with Advanced Features

// Load data on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadModelPerformance();
    loadConfusionMatrix();
    loadPRCurves();
    loadCostCurve();
    loadShapGlobal();
    loadExternalBenchmark();
    loadConceptDrift();
    loadPRTradeoff();
    loadExplanationRatings();
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
        
        // Create metrics table
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
            
            const rawName = (model.model || '').trim().toLowerCase();
            const modelName = rawName
                ? rawName.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ')
                : 'Unknown';
            
            // Highlight Random Forest as the selected model (balanced Precision & Recall)
            const isRandomForest = rawName === 'random_forest';
            if (isRandomForest) {
                row.classList.add('selected-model');
            }
            if (model.f1_score === bestF1 && !row.classList.contains('selected-model')) {
                row.classList.add('best-model');
            }
            
            // Column order: Precision, Recall, F1, PR-AUC, ROC-AUC, Accuracy (accuracy last)
            row.innerHTML = `
                <td><span class="model-name">${modelName}${isRandomForest ? ' <span class="selected-badge">Selected</span>' : ''}</span></td>
                <td>${(model.precision * 100).toFixed(2)}%</td>
                <td>${(model.recall * 100).toFixed(2)}%</td>
                <td>${(model.f1_score * 100).toFixed(2)}%</td>
                <td>${model.pr_auc != null ? (model.pr_auc * 100).toFixed(2) + '%' : 'N/A'}</td>
                <td>${model.roc_auc != null ? (model.roc_auc * 100).toFixed(2) + '%' : 'N/A'}</td>
                <td class="accuracy-cell">${(model.accuracy * 100).toFixed(2)}%</td>
            `;
            tbody.appendChild(row);
        });
        
        if (bestModel && !bestModel.classList.contains('selected-model')) {
            bestModel.classList.add('best-model');
        }
        
        // Create metrics bar chart
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
                    },
                    {
                        label: 'ROC-AUC',
                        data: models.map(m => (m.roc_auc != null ? m.roc_auc : 0) * 100),
                        backgroundColor: 'rgba(99, 102, 241, 0.8)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'PR-AUC',
                        data: models.map(m => (m.pr_auc != null ? m.pr_auc : 0) * 100),
                        backgroundColor: 'rgba(52, 211, 153, 0.7)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'Accuracy',
                        data: models.map(m => m.accuracy * 100),
                        backgroundColor: 'rgba(156, 163, 175, 0.6)',
                        borderColor: 'rgba(156, 163, 175, 1)',
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
                        text: 'ML Classifier Metrics: Precision, Recall, F1 (primary)',
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

        // Populate cost-sensitive table (expected financial loss)
        const costBody = document.getElementById('costTableBody');
        if (costBody) {
            costBody.innerHTML = '';
            models.forEach(model => {
                const row = document.createElement('tr');
                const rawNameInner = (model.model || '').trim().toLowerCase();
                const modelNameInner = rawNameInner
                    ? rawNameInner.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ')
                    : 'Unknown';
                const fnCost = model.fn_cost != null ? model.fn_cost : null;
                const fpCost = model.fp_cost != null ? model.fp_cost : null;
                const costPerTxn = model.expected_cost_per_txn != null ? model.expected_cost_per_txn : null;
                const costPerK = model.expected_cost_per_1000 != null
                    ? model.expected_cost_per_1000
                    : (costPerTxn != null ? costPerTxn * 1000.0 : null);

                row.innerHTML = `
                    <td><span class="model-name">${modelNameInner}</span></td>
                    <td>${fnCost != null ? '₹' + fnCost.toFixed(0) : '-'}</td>
                    <td>${fpCost != null ? '₹' + fpCost.toFixed(0) : '-'}</td>
                    <td>${costPerTxn != null ? '₹' + costPerTxn.toFixed(2) : '-'}</td>
                    <td>${costPerK != null ? '₹' + costPerK.toFixed(2) : '-'}</td>
                `;
                costBody.appendChild(row);
            });
        }
        
    } catch (error) {
        console.error('Error loading model performance:', error);
    }
}

// Load confusion matrix for selected model
async function loadConfusionMatrix() {
    try {
        const response = await fetch('/api/confusion_matrix');
        const data = await response.json();
        const container = document.getElementById('confusionMatrixContainer');
        const grid = document.getElementById('confusionMatrixGrid');
        const modelSpan = document.getElementById('confusionMatrixModel');
        if (!data || !data.matrix) {
            container.style.display = 'none';
            return;
        }
        const m = data.matrix;
        const modelName = (data.model || 'unknown').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        modelSpan.textContent = '(' + modelName + ')';
        grid.innerHTML = `
            <table class="confusion-matrix-table">
                <thead>
                    <tr><th></th><th>Predicted Legit</th><th>Predicted Fraud</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <th>Actual Legit</th>
                        <td class="cm-cell tn" title="True Negatives">${m[0][0]}</td>
                        <td class="cm-cell fp" title="False Positives">${m[0][1]}</td>
                    </tr>
                    <tr>
                        <th>Actual Fraud</th>
                        <td class="cm-cell fn" title="False Negatives">${m[1][0]}</td>
                        <td class="cm-cell tp" title="True Positives">${m[1][1]}</td>
                    </tr>
                </tbody>
            </table>
        `;
        container.style.display = 'block';
    } catch (error) {
        console.error('Error loading confusion matrix:', error);
    }
}

// Load Precision–Recall curves for all models
async function loadPRCurves() {
    try {
        const response = await fetch('/api/pr_curves');
        const payload = await response.json();
        if (!payload || !payload.curves || !payload.curves.length) {
            return;
        }
        const curves = payload.curves;
        const canvas = document.getElementById('prCurveChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const colorPalette = [
            'rgba(16, 185, 129, 1)',
            'rgba(245, 158, 11, 1)',
            'rgba(239, 68, 68, 1)',
            'rgba(59, 130, 246, 1)',
            'rgba(147, 51, 234, 1)',
        ];

        const datasets = curves.map((curve, idx) => {
            const color = colorPalette[idx % colorPalette.length];
            const rec = curve.recall || [];
            const prec = curve.precision || [];
            const points = rec.map((r, i) => ({
                x: r * 100.0,
                y: (prec[i] || 0) * 100.0,
            }));
            const name = (curve.model || 'model').toString().replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            return {
                label: `${name} (PR-AUC ${(curve.pr_auc * 100).toFixed(1)}%)`,
                data: points,
                borderColor: color,
                backgroundColor: color.replace('1)', '0.1)'),
                fill: false,
                tension: 0.2,
            };
        });

        new Chart(ctx, {
            type: 'line',
            data: {
                datasets: datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    title: {
                        display: true,
                        text: 'Precision–Recall Curve (All Models)',
                        font: { size: 14, weight: 'bold' },
                    },
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Recall (%)',
                        },
                        min: 0,
                        max: 100,
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Precision (%)',
                        },
                        min: 0,
                        max: 100,
                    },
                },
            },
        });
    } catch (error) {
        console.error('Error loading PR curves:', error);
    }
}

// Load cost vs threshold curve for selected model
async function loadCostCurve() {
    try {
        const response = await fetch('/api/cost_curve');
        const data = await response.json();
        if (!data || !data.thresholds || !data.total_cost_per_1000) {
            return;
        }
        const canvas = document.getElementById('costThresholdChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const thresholds = data.thresholds.map(t => Number(t.toFixed(2)));
        const costs = data.total_cost_per_1000.map(c => Number(c.toFixed(2)));

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: thresholds,
                datasets: [
                    {
                        label: 'Expected Cost per 1,000 Txns',
                        data: costs,
                        borderColor: 'rgba(239, 68, 68, 1)',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                    },
                ],
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
                        text: 'Cost vs Decision Threshold (Selected Model)',
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Decision Threshold',
                        },
                        min: 0,
                        max: 1,
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Expected Cost per 1,000 Transactions',
                        },
                        beginAtZero: true,
                    },
                },
            },
        });
    } catch (error) {
        console.error('Error loading cost curve:', error);
    }
}

// Load global SHAP importance (Random Forest)
async function loadShapGlobal() {
    try {
        const response = await fetch('/api/shap_global');
        const data = await response.json();
        if (!data || !data.features || !data.importance) {
            return;
        }
        const canvas = document.getElementById('shapGlobalChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const labels = data.features;
        const imp = data.importance;

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Mean |SHAP| (global importance)',
                    data: imp,
                    backgroundColor: 'rgba(139, 92, 246, 0.8)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'Global SHAP Feature Importance (Fraud class)',
                    },
                },
                scales: {
                    x: { beginAtZero: true },
                },
            },
        });
    } catch (error) {
        console.error('Error loading global SHAP:', error);
    }
}

// Local SHAP explanation (per-transaction) is currently disabled for stability.

// Load external dataset benchmark results (if available)
async function loadExternalBenchmark() {
    try {
        const response = await fetch('/api/external_benchmark');
        const data = await response.json();
        const tbody = document.getElementById('externalBenchmarkBody');
        if (!tbody) return;

        tbody.innerHTML = '';
        if (!data || data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="loading">No external benchmark results found. Place datasets in <code>data/</code> and run training.</td></tr>';
            return;
        }

        data.forEach(row => {
            const tr = document.createElement('tr');
            const fraudRate = row.fraud_rate != null ? (Number(row.fraud_rate) * 100).toFixed(2) + '%' : '-';
            const f1 = row.f1_score != null ? (Number(row.f1_score) * 100).toFixed(2) + '%' : '-';
            const roc = row.roc_auc != null ? (Number(row.roc_auc) * 100).toFixed(2) + '%' : '-';
            tr.innerHTML = `
                <td>${row.dataset || '-'}</td>
                <td>${row.samples != null ? Number(row.samples).toLocaleString() : '-'}</td>
                <td>${fraudRate}</td>
                <td>${f1}</td>
                <td>${roc}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        console.error('Error loading external benchmark:', error);
        const tbody = document.getElementById('externalBenchmarkBody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="5" class="loading">Error loading external benchmark.</td></tr>';
        }
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
        // Get the original prediction to access transaction_data
        const recentResponse = await fetch('/api/recent_predictions');
        const recentPredictions = await recentResponse.json();
        const originalPred = recentPredictions.find(p => p.id === predictionId);
        
        if (!originalPred || !originalPred.transaction_data) {
            alert('Could not find original prediction data.');
            return;
        }
        
        // Compute SHAP explanation
        let shapData = null;
        try {
            const shapResponse = await fetch('/api/compute_shap_local', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...originalPred.transaction_data,
                    is_fraud: originalPred.risk_level === 'High' ? 1 : 0
                })
            });
            if (shapResponse.ok) {
                shapData = await shapResponse.json();
            }
        } catch (e) {
            console.warn('Could not compute SHAP:', e);
        }
        
        // Get LLM explanation
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
        
        // Show comparison if both SHAP and LLM are available
        if (shapData && result.explanation) {
            showExplanationComparison(
                shapData,
                result.explanation,
                result.risk_factors || [],
                originalPred.transaction_data
            );
        }
        
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
            ${(explanation || riskFactors.length > 0) ? `
                <div class="explanation-section ${isRisky ? 'risky' : ''}">
                    <h4 class="explanation-section-title">Decision Rationale</h4>
                    ${explanation ? `
                        <div class="explanation-box">
                            <p>${String(explanation).replace(/\n/g, '<br>')}</p>
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

// Load Concept Drift Simulation
async function loadConceptDrift() {
    try {
        const response = await fetch('/api/concept_drift');
        const data = await response.json();
        
        if (!data || !data.drift_performance) {
            document.getElementById('driftDescription').textContent = 'Concept drift simulation not available. Run training to generate.';
            return;
        }
        
        const periods = data.drift_performance.map(d => d.period);
        const accuracy = data.drift_performance.map(d => d.accuracy);
        const precision = data.drift_performance.map(d => d.precision);
        const recall = data.drift_performance.map(d => d.recall);
        const f1 = data.drift_performance.map(d => d.f1_score);
        
        const ctx = document.getElementById('conceptDriftChart').getContext('2d');
        if (window.conceptDriftChart) {
            window.conceptDriftChart.destroy();
        }
        
        window.conceptDriftChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: periods,
                datasets: [
                    {
                        label: 'Accuracy',
                        data: accuracy,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    },
                    {
                        label: 'Precision',
                        data: precision,
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    },
                    {
                        label: 'Recall',
                        data: recall,
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    },
                    {
                        label: 'F1-Score',
                        data: f1,
                        borderColor: 'rgb(255, 206, 86)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Degradation Over Time (Concept Drift)'
                    },
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0,
                        max: 1
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time Period'
                        }
                    }
                }
            }
        });
        
        const initialF1 = f1[0];
        const finalF1 = f1[f1.length - 1];
        const degradation = ((initialF1 - finalF1) / initialF1 * 100).toFixed(1);
        document.getElementById('driftDescription').textContent = 
            `F1-Score degradation: ${degradation}% (from ${initialF1.toFixed(3)} to ${finalF1.toFixed(3)}) over ${periods.length} periods.`;
    } catch (error) {
        console.error('Error loading concept drift:', error);
        document.getElementById('driftDescription').textContent = 'Error loading concept drift data.';
    }
}

// Load PR Tradeoff Comparison
async function loadPRTradeoff() {
    try {
        const response = await fetch('/api/pr_tradeoff');
        const data = await response.json();
        
        if (!data || !data.thresholds) {
            document.getElementById('prTradeoffBody').innerHTML = 
                '<tr><td colspan="5" class="loading">PR tradeoff data not available. Run training to generate.</td></tr>';
            return;
        }
        
        // Update table
        const tbody = document.getElementById('prTradeoffBody');
        tbody.innerHTML = '';
        data.thresholds.forEach((thresh, idx) => {
            const rf = data.random_forest[idx];
            const xgb = data.xgboost[idx];
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${thresh.toFixed(2)}</td>
                <td>${rf.precision.toFixed(3)}</td>
                <td>${rf.recall.toFixed(3)}</td>
                <td>${xgb.precision.toFixed(3)}</td>
                <td>${xgb.recall.toFixed(3)}</td>
            `;
            tbody.appendChild(row);
        });
        
        // Create chart
        const ctx = document.getElementById('prTradeoffChart').getContext('2d');
        if (window.prTradeoffChart) {
            window.prTradeoffChart.destroy();
        }
        
        window.prTradeoffChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.thresholds.map(t => t.toFixed(2)),
                datasets: [
                    {
                        label: 'RF Precision',
                        data: data.random_forest.map(r => r.precision),
                        borderColor: 'rgb(255, 99, 132)',
                        yAxisID: 'y',
                        tension: 0.1
                    },
                    {
                        label: 'RF Recall',
                        data: data.random_forest.map(r => r.recall),
                        borderColor: 'rgb(255, 99, 132)',
                        borderDash: [5, 5],
                        yAxisID: 'y',
                        tension: 0.1
                    },
                    {
                        label: 'XGBoost Precision',
                        data: data.xgboost.map(r => r.precision),
                        borderColor: 'rgb(54, 162, 235)',
                        yAxisID: 'y',
                        tension: 0.1
                    },
                    {
                        label: 'XGBoost Recall',
                        data: data.xgboost.map(r => r.recall),
                        borderColor: 'rgb(54, 162, 235)',
                        borderDash: [5, 5],
                        yAxisID: 'y',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Precision-Recall Tradeoff: RF vs XGBoost'
                    },
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Precision / Recall'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Threshold'
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading PR tradeoff:', error);
        document.getElementById('prTradeoffBody').innerHTML = 
            '<tr><td colspan="5" class="loading">Error loading PR tradeoff data.</td></tr>';
    }
}

// Load Explanation Ratings
async function loadExplanationRatings() {
    try {
        const response = await fetch('/api/explanation_ratings');
        const data = await response.json();
        
        const shapAvg = data.averages.shap || 0;
        const llmAvg = data.averages.llm || 0;
        const shapCount = data.all_ratings.filter(r => r.explanation_type === 'shap').length;
        const llmCount = data.all_ratings.filter(r => r.explanation_type === 'llm').length;
        
        document.getElementById('shapAvgRating').textContent = shapAvg.toFixed(2);
        document.getElementById('llmAvgRating').textContent = llmAvg.toFixed(2);
        document.getElementById('shapRatingCount').textContent = `(${shapCount} ratings)`;
        document.getElementById('llmRatingCount').textContent = `(${llmCount} ratings)`;
    } catch (error) {
        console.error('Error loading explanation ratings:', error);
    }
}

// Submit Rating
async function submitRating() {
    try {
        const explanationType = document.getElementById('ratingType').value;
        const transactionId = document.getElementById('ratingTransactionId').value || '';
        const rating = parseInt(document.getElementById('ratingValue').value);
        const comments = document.getElementById('ratingComments').value;
        
        const response = await fetch('/api/explanation_rating', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                explanation_type: explanationType,
                transaction_id: transactionId || 'manual_' + Date.now(),
                rating: rating,
                comments: comments
            })
        });
        
        if (response.ok) {
            alert('Rating submitted successfully!');
            document.getElementById('ratingTransactionId').value = '';
            document.getElementById('ratingComments').value = '';
            document.getElementById('ratingValue').value = '3';
            loadExplanationRatings();
        } else {
            const error = await response.json();
            alert('Error: ' + (error.error || 'Failed to submit rating'));
        }
    } catch (error) {
        console.error('Error submitting rating:', error);
        alert('Error submitting rating. Please try again.');
    }
}

// Show Explanation Comparison (called when LLM explanation is generated)
async function showExplanationComparison(shapData, llmReasoning, llmRiskFactors, transactionData) {
    try {
        const response = await fetch('/api/explanation_comparison', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                shap_data: shapData,
                llm_reasoning: llmReasoning,
                llm_risk_factors: llmRiskFactors,
                transaction_data: transactionData
            })
        });
        
        if (!response.ok) {
            console.error('Failed to compare explanations');
            return;
        }
        
        const comparison = await response.json();
        
        // Display SHAP explanation
        const shapContent = document.getElementById('shapComparisonContent');
        const shapTop = comparison.shap_explanation.top_features.slice(0, 10);
        shapContent.innerHTML = `
            <div class="shap-features">
                <p><strong>Predicted Probability:</strong> ${comparison.shap_explanation.predicted_proba.toFixed(3)}</p>
                <p><strong>Base Value:</strong> ${comparison.shap_explanation.base_value.toFixed(3)}</p>
                <h4>Top Contributing Features:</h4>
                <ul>
                    ${shapTop.map(f => `
                        <li>
                            <strong>${f.feature}:</strong> 
                            <span style="color: ${f.shap_value > 0 ? '#e74c3c' : '#27ae60'}">
                                ${f.shap_value > 0 ? '+' : ''}${f.shap_value.toFixed(4)}
                            </span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
        
        // Display LLM explanation
        const llmContent = document.getElementById('llmComparisonContent');
        llmContent.innerHTML = `
            <div class="llm-explanation">
                <p><strong>Reasoning:</strong></p>
                <p>${comparison.llm_explanation.reasoning.replace(/\n/g, '<br>')}</p>
                <h4>Risk Factors:</h4>
                <ul>
                    ${comparison.llm_explanation.risk_factors.map(f => `<li>${f}</li>`).join('')}
                </ul>
                <p><strong>Mentioned Features:</strong> ${comparison.llm_explanation.mentioned_features.join(', ') || 'None'}</p>
            </div>
        `;
        
        // Display alignment score
        document.getElementById('alignmentScore').textContent = 
            `${(comparison.comparison.alignment_score * 100).toFixed(1)}% (${comparison.comparison.alignment_count}/${comparison.comparison.shap_top_count} features aligned)`;
        
        document.getElementById('comparisonContent').style.display = 'block';
    } catch (error) {
        console.error('Error showing explanation comparison:', error);
    }
}
