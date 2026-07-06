/**
 * Risk Calculator Engine - Interactive Logistic Regression Calculator
 * 
 * Fetches trained machine learning assets, binds form sliders and selectors,
 * standardizes inputs, and predicts the probability of High Burnout.
 */

document.addEventListener('DOMContentLoaded', () => {
    let modelAssets = null;

    // Feature labels for explainability view
    const featureLabels = {
        "age": "Age",
        "year": "Academic Year",
        "daily_study_hours": "Study Hours",
        "daily_sleep_hours": "Daily Sleep",
        "screen_time_hours": "Screen Time",
        "stress_level": "Stress Level",
        "anxiety_score": "Anxiety Score",
        "depression_score": "Depression Score",
        "academic_pressure_score": "Academic Pressure",
        "financial_stress_score": "Financial Stress",
        "social_support_score": "Social Support",
        "physical_activity_hours": "Exercise Duration",
        "sleep_quality": "Sleep Quality",
        "attendance_percentage": "Attendance Rate",
        "cgpa": "Academic CGPA",
        "internet_quality": "Internet Quality",
        "gender_Male": "Male Gender",
        "gender_Other": "Non-binary Gender",
        "course_BCA": "BCA Course",
        "course_BSc": "BSc Course",
        "course_BTech": "BTech Course",
        "course_MBA": "MBA Course",
        "course_MCA": "MCA Course"
    };

    // Load assets asynchronously
    async function init() {
        try {
            console.log("Loading model assets dynamically...");
            const response = await fetch(`model_assets.json?v=${Date.now()}`);
            if (!response.ok) {
                throw new Error(`Failed to retrieve model weights. Status: ${response.status}`);
            }
            modelAssets = await response.json();
            console.log("Model assets loaded successfully:", modelAssets.model_name);

            setupEventListeners();
            calculateRisk();
        } catch (error) {
            console.error("Calculator initialization failed:", error);
            displayInitializationError(error.message);
        }
    }

    // Displays warning message on interface in case of fetch failures
    function displayInitializationError(message) {
        const descEl = document.getElementById('riskDesc');
        if (descEl) {
            descEl.innerHTML = `<span style="color:var(--danger)">⚠️ <strong>System Error:</strong> Unable to load machine learning model weights. ${message}. Ensure you are serving the page via a web server.</span>`;
        }
        const badgeEl = document.getElementById('riskBadge');
        if (badgeEl) {
            badgeEl.className = 'badge badge-high';
            badgeEl.innerText = 'OFFLINE';
        }
        const pctEl = document.getElementById('riskPct');
        if (pctEl) pctEl.innerText = '--%';
    }

    // Binds DOM events
    function setupEventListeners() {
        const formInputs = document.querySelectorAll('#burnoutForm input, #burnoutForm select');
        
        // Initialize range slider background tracks
        formInputs.forEach(input => {
            if (input.type === 'range') {
                updateSliderTrack(input);
            }
        });

        // Binds slider movements and selection shifts
        formInputs.forEach(input => {
            input.addEventListener('input', () => {
                const labelVal = document.getElementById(input.id + '_val');
                if (labelVal) {
                    let suffix = '';
                    if (input.id.includes('hours') || input.id === 'daily_sleep_hours') suffix = ' hrs';
                    else if (input.id.includes('percentage')) suffix = '%';
                    else if (input.id.includes('score')) suffix = ' / 10';
                    else if (input.id === 'age') suffix = ' yrs';
                    
                    labelVal.innerText = input.value + suffix;
                }
                
                if (input.type === 'range') {
                    updateSliderTrack(input);
                }
                
                calculateRisk();
            });
        });

        // Binds reset triggers
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                document.getElementById('burnoutForm').reset();
                formInputs.forEach(input => {
                    const labelVal = document.getElementById(input.id + '_val');
                    if (labelVal) {
                        let suffix = '';
                        if (input.id.includes('hours') || input.id === 'daily_sleep_hours') suffix = ' hrs';
                        else if (input.id.includes('percentage')) suffix = '%';
                        else if (input.id.includes('score')) suffix = ' / 10';
                        else if (input.id === 'age') suffix = ' yrs';
                        labelVal.innerText = input.value + suffix;
                    }
                    if (input.type === 'range') {
                        updateSliderTrack(input);
                    }
                });
                calculateRisk();
            });
        }
    }

    // Sync slider fill color percentage with thumb coordinate
    function updateSliderTrack(slider) {
        if (slider.type !== 'range') return;
        const min = parseFloat(slider.min) || 0;
        const max = parseFloat(slider.max) || 100;
        const val = parseFloat(slider.value) || 0;
        const percent = ((val - min) / (max - min)) * 100;
        slider.style.setProperty('--value-percent', `${percent}%`);
    }

    // Gathers inputs, executes scaling, calculates logistic regression outcome
    function calculateRisk() {
        if (!modelAssets) return;

        try {
            const values = {};
            
            // 1. Safe parsing and boundaries validation for numerical inputs
            const parseAndClamp = (id, min, max, def) => {
                const el = document.getElementById(id);
                if (!el) return def;
                let val = parseFloat(el.value);
                if (isNaN(val)) return def;
                return Math.max(min, Math.min(max, val));
            };

            values['age'] = parseAndClamp('age', 17, 25, 21);
            values['daily_study_hours'] = parseAndClamp('daily_study_hours', 1.0, 10.0, 5.5);
            values['daily_sleep_hours'] = parseAndClamp('daily_sleep_hours', 1.0, 10.0, 6.5);
            values['screen_time_hours'] = parseAndClamp('screen_time_hours', 0.0, 12.0, 6.5);
            values['anxiety_score'] = parseAndClamp('anxiety_score', 0, 10, 5);
            values['depression_score'] = parseAndClamp('depression_score', 0, 10, 5);
            values['academic_pressure_score'] = parseAndClamp('academic_pressure_score', 0, 10, 5);
            values['financial_stress_score'] = parseAndClamp('financial_stress_score', 0, 10, 5);
            values['social_support_score'] = parseAndClamp('social_support_score', 0, 10, 5);
            values['physical_activity_hours'] = parseAndClamp('physical_activity_hours', 0.0, 3.0, 1.0);
            values['attendance_percentage'] = parseAndClamp('attendance_percentage', 50, 100, 75);
            values['cgpa'] = parseAndClamp('cgpa', 4.0, 10.0, 7.0);

            // 2. Map ordinal parameters safely
            const getMappedVal = (id, mappingKey, defKey) => {
                const el = document.getElementById(id);
                const valStr = el ? el.value : defKey;
                const mapping = modelAssets.categorical_mappings[mappingKey];
                return mapping && mapping[valStr] !== undefined ? mapping[valStr] : mapping[defKey];
            };

            values['year'] = getMappedVal('year', 'year', '2nd');
            values['stress_level'] = getMappedVal('stress_level', 'stress_level', 'Medium');
            values['sleep_quality'] = getMappedVal('sleep_quality', 'sleep_quality', 'Average');
            values['internet_quality'] = getMappedVal('internet_quality', 'internet_quality', 'Average');

            // 3. One-hot categories (Gender & Course)
            const genderEl = document.getElementById('gender');
            const genderVal = genderEl ? genderEl.value : 'Male';
            values['gender_Male'] = (genderVal === 'Male') ? 1.0 : 0.0;
            values['gender_Other'] = (genderVal === 'Other') ? 1.0 : 0.0;

            const courseEl = document.getElementById('course');
            const courseVal = courseEl ? courseEl.value : 'BSc';
            values['course_BCA'] = (courseVal === 'BCA') ? 1.0 : 0.0;
            values['course_BSc'] = (courseVal === 'BSc') ? 1.0 : 0.0;
            values['course_BTech'] = (courseVal === 'BTech') ? 1.0 : 0.0;
            values['course_MBA'] = (courseVal === 'MBA') ? 1.0 : 0.0;
            values['course_MCA'] = (courseVal === 'MCA') ? 1.0 : 0.0;

            // 4. Compute Standard Scaled values and dot product (z score)
            let z = modelAssets.intercept;
            const contributions = [];

            for (const feature in modelAssets.coefficients) {
                const scaling = modelAssets.scaling[feature];
                if (!scaling) continue;

                const mean = scaling.mean;
                const std = scaling.std;
                const coef = modelAssets.coefficients[feature];

                const rawValue = values[feature];
                let scaledValue = 0.0; // Default fallback to scaler mean (z-score of 0.0)
                
                if (rawValue !== undefined && std > 0) {
                    scaledValue = (rawValue - mean) / std;
                } else if (rawValue === undefined) {
                    console.warn(`Feature ${feature} missing from DOM inputs. Falling back to mean value.`);
                }

                const contribution = coef * scaledValue;
                
                z += contribution;
                
                contributions.push({
                    name: feature,
                    label: featureLabels[feature] || feature,
                    val: contribution,
                    coef: coef,
                    scaledVal: scaledValue
                });
            }

            // 5. Sigmoid linked function: P = 1 / (1 + e^-z)
            const probability = 1 / (1 + Math.exp(-z));
            const probabilityPct = Math.round(probability * 100);

            // 6. Update progress bar visual elements
            const progressBar = document.getElementById('progressBar');
            if (progressBar) {
                const circumference = 2 * Math.PI * 70; // r = 70
                const offset = circumference - (probability * circumference);
                progressBar.style.strokeDashoffset = offset;
            }
            
            const pctEl = document.getElementById('riskPct');
            if (pctEl) pctEl.innerText = probabilityPct + '%';

            // 7. Update classification status badge
            const badge = document.getElementById('riskBadge');
            const label = document.getElementById('riskLabel');
            const desc = document.getElementById('riskDesc');

            if (badge && label && desc) {
                if (probability < 0.33) {
                    badge.className = 'badge badge-low';
                    badge.innerText = 'Low Risk';
                    label.innerText = 'Low Risk';
                    if (progressBar) progressBar.style.stroke = 'var(--success)';
                    desc.innerHTML = 'This profile reflects a <strong>low vulnerability</strong> to academic burnout. Maintaining healthy lifestyle habits, academic boundaries, and consistent rest will keep burnout risks low.';
                } else if (probability < 0.66) {
                    badge.className = 'badge badge-medium';
                    badge.innerText = 'Moderate Risk';
                    label.innerText = 'Mod. Risk';
                    if (progressBar) progressBar.style.stroke = 'var(--warning)';
                    desc.innerHTML = 'This profile reflects a <strong>moderate risk</strong> of academic burnout. Consider optimizing sleep patterns, limiting screen time, and adjusting study loads before symptoms escalate.';
                } else {
                    badge.className = 'badge badge-high';
                    badge.innerText = 'High Risk';
                    label.innerText = 'High Risk';
                    if (progressBar) progressBar.style.stroke = 'var(--danger)';
                    desc.innerHTML = 'This profile indicates a <strong>high vulnerability</strong> to academic burnout. We strongly recommend reducing academic pressure, boosting social support networks, and prioritizing quality sleep.';
                }
            }

            // 8. Render explainability factors (Top 2 risk vs protective)
            contributions.sort((a, b) => b.val - a.val);

            const riskFactors = contributions.filter(c => c.val > 0).slice(0, 2);
            const protectiveFactors = contributions.filter(c => c.val < 0).slice(-2).reverse();

            const factorList = document.getElementById('factorList');
            if (factorList) {
                factorList.innerHTML = '';

                // Risk elements
                riskFactors.forEach(factor => {
                    const item = document.createElement('div');
                    item.className = 'factor-item';
                    item.innerHTML = `
                        <span class="factor-name factor-risk">
                            <svg width="12" height="12" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
                            ${factor.label} (Elevated)
                        </span>
                        <span class="factor-weight">+${Math.abs(factor.val).toFixed(2)}</span>
                    `;
                    factorList.appendChild(item);
                });

                // Protective elements
                protectiveFactors.forEach(factor => {
                    const item = document.createElement('div');
                    item.className = 'factor-item';
                    item.innerHTML = `
                        <span class="factor-name factor-protective">
                            <svg width="12" height="12" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>
                            ${factor.label} (Favorable)
                        </span>
                        <span class="factor-weight">-${Math.abs(factor.val).toFixed(2)}</span>
                    `;
                    factorList.appendChild(item);
                });
                
                if (riskFactors.length === 0 && protectiveFactors.length === 0) {
                    factorList.innerHTML = '<div style="font-size:0.8rem;color:var(--text-secondary)">No dominant factors.</div>';
                }
            }
        } catch (err) {
            console.error("Risk calculation error:", err);
        }
    }

    // Launch calculator
    init();
});
