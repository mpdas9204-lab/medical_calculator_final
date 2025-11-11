import streamlit as st
import numpy as np
from scipy import stats
import pandas as pd
from math import ceil, sqrt, log

# Page configuration
st.set_page_config(
    page_title="Medical Sample Size Calculator",
    page_icon="🏥",)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-box {
        background-color: #f0f8e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
    }
    .result-box {
        background-color: #fff4e6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'topic' not in st.session_state:
    st.session_state.topic = ""
if 'research_question' not in st.session_state:
    st.session_state.research_question = ""
if 'study_type' not in st.session_state:
    st.session_state.study_type = ""

# Title
st.markdown('<h1 class="main-header">🏥 Medical Sample Size Calculator</h1>', unsafe_allow_html=True)
st.markdown("### A Comprehensive Tool for Medical Students and Researchers")

# Study type recommendation logic
def recommend_study_type(research_question, available_time, available_resources):
    """
    Recommends appropriate study design based on research question and constraints
    """
    recommendations = []
    
    # Keywords for different study types
    prevalence_keywords = ['prevalence', 'frequency', 'how common', 'how many', 'proportion', 'rate']
    association_keywords = ['association', 'relationship', 'correlation', 'risk factor', 'related to']
    causation_keywords = ['cause', 'effect', 'treatment', 'intervention', 'efficacy', 'outcome']
    diagnostic_keywords = ['diagnostic', 'sensitivity', 'specificity', 'accuracy', 'test']
    prognosis_keywords = ['prognosis', 'survival', 'outcome', 'follow-up', 'predict']
    
    question_lower = research_question.lower()
    
    # Prevalence studies
    if any(keyword in question_lower for keyword in prevalence_keywords):
        recommendations.append({
            'type': 'Cross-sectional Study',
            'score': 9,
            'reason': 'Best for determining prevalence and frequency of conditions at a single point in time',
            'pros': ['Quick to conduct', 'Cost-effective', 'Good for prevalence estimation'],
            'cons': ['Cannot establish causality', 'Temporal relationship unclear']
        })
    
    # Association studies
    if any(keyword in question_lower for keyword in association_keywords):
        recommendations.append({
            'type': 'Case-Control Study',
            'score': 8,
            'reason': 'Efficient for studying associations, especially for rare diseases',
            'pros': ['Good for rare diseases', 'Faster than cohort studies', 'Less expensive'],
            'cons': ['Recall bias', 'Cannot calculate incidence', 'Selection bias risk']
        })
        recommendations.append({
            'type': 'Cohort Study',
            'score': 7,
            'reason': 'Can establish temporal relationships and calculate incidence',
            'pros': ['Can calculate incidence', 'Multiple outcomes', 'Less bias than case-control'],
            'cons': ['Time-consuming', 'Expensive', 'Loss to follow-up']
        })
    
    # Causation/Treatment studies
    if any(keyword in question_lower for keyword in causation_keywords):
        recommendations.append({
            'type': 'Randomized Controlled Trial (RCT)',
            'score': 10,
            'reason': 'Gold standard for establishing causation and treatment efficacy',
            'pros': ['Highest level of evidence', 'Controls confounding', 'Establishes causality'],
            'cons': ['Expensive', 'Time-consuming', 'Ethical constraints', 'May lack external validity']
        })
        recommendations.append({
            'type': 'Cohort Study',
            'score': 7,
            'reason': 'Alternative when RCT is not feasible',
            'pros': ['More ethical for harmful exposures', 'Real-world data'],
            'cons': ['Cannot control allocation', 'Confounding possible']
        })
    
    # Diagnostic studies
    if any(keyword in question_lower for keyword in diagnostic_keywords):
        recommendations.append({
            'type': 'Diagnostic Accuracy Study',
            'score': 10,
            'reason': 'Specifically designed to evaluate diagnostic tests',
            'pros': ['Direct assessment of test performance', 'Calculates sensitivity/specificity'],
            'cons': ['Needs gold standard', 'Spectrum bias risk']
        })
    
    # Prognosis studies
    if any(keyword in question_lower for keyword in prognosis_keywords):
        recommendations.append({
            'type': 'Cohort Study',
            'score': 9,
            'reason': 'Best for studying outcomes over time',
            'pros': ['Natural history observation', 'Multiple endpoints possible'],
            'cons': ['Long duration', 'Loss to follow-up']
        })
    
    # Adjust scores based on resources and time
    if available_time == "Short (< 6 months)":
        for rec in recommendations:
            if rec['type'] in ['Cross-sectional Study', 'Case-Control Study']:
                rec['score'] += 2
            elif rec['type'] in ['RCT', 'Cohort Study']:
                rec['score'] -= 2
    
    if available_resources == "Limited":
        for rec in recommendations:
            if rec['type'] in ['Cross-sectional Study', 'Case-Control Study']:
                rec['score'] += 1
            elif rec['type'] == 'Randomized Controlled Trial (RCT)':
                rec['score'] -= 3
    
    # Sort by score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    # If no specific recommendations, provide general guidance
    if not recommendations:
        recommendations = [
            {
                'type': 'Cross-sectional Study',
                'score': 5,
                'reason': 'General purpose study for initial exploration',
                'pros': ['Quick', 'Inexpensive', 'Easy to conduct'],
                'cons': ['Limited causal inference']
            }
        ]
    
    return recommendations

# Sample size calculation functions
def calculate_cross_sectional(expected_prevalence, confidence_level, margin_error, dropout_rate):
    """Calculate sample size for cross-sectional study"""
    z_score = stats.norm.ppf((1 + confidence_level/100) / 2)
    p = expected_prevalence / 100
    e = margin_error / 100
    
    n = (z_score**2 * p * (1-p)) / e**2
    n_adjusted = n / (1 - dropout_rate/100)
    
    return ceil(n_adjusted), ceil(n)

def calculate_case_control(power, confidence_level, odds_ratio, control_exposure, case_control_ratio, dropout_rate):
    """Calculate sample size for case-control study"""
    z_alpha = stats.norm.ppf((1 + confidence_level/100) / 2)
    z_beta = stats.norm.ppf(power/100)
    
    p1 = control_exposure / 100
    p2 = (odds_ratio * p1) / (1 - p1 + odds_ratio * p1)
    p_avg = (p1 + case_control_ratio * p2) / (1 + case_control_ratio)
    
    n_cases = ((z_alpha * sqrt((1 + case_control_ratio) * p_avg * (1 - p_avg)) + 
                z_beta * sqrt(case_control_ratio * p1 * (1 - p1) + p2 * (1 - p2)))**2) / \
               (case_control_ratio * (p2 - p1)**2)
    
    n_controls = n_cases * case_control_ratio
    n_cases_adjusted = ceil(n_cases / (1 - dropout_rate/100))
    n_controls_adjusted = ceil(n_controls / (1 - dropout_rate/100))
    
    return n_cases_adjusted, n_controls_adjusted, ceil(n_cases), ceil(n_controls)

def calculate_cohort(power, confidence_level, relative_risk, control_incidence, exposed_unexposed_ratio, dropout_rate):
    """Calculate sample size for cohort study"""
    z_alpha = stats.norm.ppf((1 + confidence_level/100) / 2)
    z_beta = stats.norm.ppf(power/100)
    
    p1 = control_incidence / 100
    p2 = p1 * relative_risk
    
    if p2 > 1:
        st.error("The expected incidence in exposed group exceeds 100%. Please adjust your parameters.")
        return None, None, None, None
    
    p_avg = (p1 + exposed_unexposed_ratio * p2) / (1 + exposed_unexposed_ratio)
    
    n_unexposed = ((z_alpha * sqrt((1 + exposed_unexposed_ratio) * p_avg * (1 - p_avg)) + 
                    z_beta * sqrt(exposed_unexposed_ratio * p1 * (1 - p1) + p2 * (1 - p2)))**2) / \
                   (exposed_unexposed_ratio * (p2 - p1)**2)
    
    n_exposed = n_unexposed * exposed_unexposed_ratio
    n_unexposed_adjusted = ceil(n_unexposed / (1 - dropout_rate/100))
    n_exposed_adjusted = ceil(n_exposed / (1 - dropout_rate/100))
    
    return n_exposed_adjusted, n_unexposed_adjusted, ceil(n_exposed), ceil(n_unexposed)

def calculate_rct(power, confidence_level, effect_size, control_mean, control_sd, allocation_ratio, dropout_rate):
    """Calculate sample size for RCT with continuous outcome"""
    z_alpha = stats.norm.ppf((1 + confidence_level/100) / 2)
    z_beta = stats.norm.ppf(power/100)
    
    delta = effect_size
    sigma = control_sd
    
    n_control = ((z_alpha + z_beta)**2 * 2 * sigma**2 * (1 + 1/allocation_ratio)) / delta**2
    n_intervention = n_control * allocation_ratio
    
    n_control_adjusted = ceil(n_control / (1 - dropout_rate/100))
    n_intervention_adjusted = ceil(n_intervention / (1 - dropout_rate/100))
    
    return n_intervention_adjusted, n_control_adjusted, ceil(n_intervention), ceil(n_control)

def calculate_rct_binary(power, confidence_level, control_event_rate, intervention_event_rate, allocation_ratio, dropout_rate):
    """Calculate sample size for RCT with binary outcome"""
    z_alpha = stats.norm.ppf((1 + confidence_level/100) / 2)
    z_beta = stats.norm.ppf(power/100)
    
    p1 = control_event_rate / 100
    p2 = intervention_event_rate / 100
    p_avg = (p1 + allocation_ratio * p2) / (1 + allocation_ratio)
    
    n_control = ((z_alpha * sqrt((1 + allocation_ratio) * p_avg * (1 - p_avg)) + 
                  z_beta * sqrt(allocation_ratio * p1 * (1 - p1) + p2 * (1 - p2)))**2) / \
                 (allocation_ratio * (p2 - p1)**2)
    
    n_intervention = n_control * allocation_ratio
    n_control_adjusted = ceil(n_control / (1 - dropout_rate/100))
    n_intervention_adjusted = ceil(n_intervention / (1 - dropout_rate/100))
    
    return n_intervention_adjusted, n_control_adjusted, ceil(n_intervention), ceil(n_control)

def calculate_diagnostic(sensitivity, specificity, prevalence, confidence_level, margin_error, dropout_rate):
    """Calculate sample size for diagnostic accuracy study"""
    z_score = stats.norm.ppf((1 + confidence_level/100) / 2)
    
    # Sample size for sensitivity
    se = sensitivity / 100
    e = margin_error / 100
    n_diseased = (z_score**2 * se * (1 - se)) / e**2
    
    # Sample size for specificity
    sp = specificity / 100
    n_non_diseased = (z_score**2 * sp * (1 - sp)) / e**2
    
    # Total sample size based on prevalence
    prev = prevalence / 100
    n_total = max(n_diseased / prev, n_non_diseased / (1 - prev))
    
    n_total_adjusted = ceil(n_total / (1 - dropout_rate/100))
    
    return n_total_adjusted, ceil(n_total), ceil(n_diseased), ceil(n_non_diseased)

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Step 1: Research Topic
st.sidebar.markdown("### 📝 Step 1: Research Topic")
if st.sidebar.button("Go to Step 1", key="nav_step1"):
    st.session_state.step = 1

# Step 2: Study Design Recommendation
st.sidebar.markdown("### 🔍 Step 2: Study Design")
if st.sidebar.button("Go to Step 2", key="nav_step2", disabled=not st.session_state.topic):
    st.session_state.step = 2

# Step 3: Sample Size Calculation
st.sidebar.markdown("### 📊 Step 3: Sample Size")
if st.sidebar.button("Go to Step 3", key="nav_step3", disabled=not st.session_state.study_type):
    st.session_state.step = 3

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Complete each step in order for best results!")

# Main content based on step
if st.session_state.step == 1:
    # STEP 1: Research Topic Input
    st.markdown('<h2 class="sub-header">Step 1: Define Your Research Topic</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Instructions:**
        - Enter your research topic or area of interest
        - Describe your research question clearly
        - This will help us recommend the most appropriate study design
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        topic = st.text_input(
            "Research Topic/Area:",
            value=st.session_state.topic,
            placeholder="e.g., Diabetes, Hypertension, COVID-19, Cancer screening",
            help="Enter the main medical condition or area you're studying"
        )
        
        research_question = st.text_area(
            "Research Question:",
            value=st.session_state.research_question,
            placeholder="e.g., What is the prevalence of diabetes in adults aged 40-60?\nIs smoking associated with lung cancer?\nDoes drug X reduce blood pressure better than drug Y?",
            help="Be specific about what you want to investigate",
            height=150
        )
        
        if st.button("Continue to Study Design Recommendation →", type="primary", disabled=not (topic and research_question)):
            st.session_state.topic = topic
            st.session_state.research_question = research_question
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        st.markdown("### 📚 Example Questions")
        st.markdown("""
        **Prevalence:**
        - What is the prevalence of...?
        - How common is...?
        
        **Association:**
        - Is X associated with Y?
        - What are the risk factors for...?
        
        **Causation:**
        - Does treatment X cause improvement in...?
        - What is the effect of intervention Y on...?
        
        **Diagnosis:**
        - What is the accuracy of test X for detecting...?
        - How sensitive/specific is...?
        """)

elif st.session_state.step == 2:
    # STEP 2: Study Design Recommendation
    st.markdown('<h2 class="sub-header">Step 2: Study Design Recommendation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Research Topic:** {st.session_state.topic}")
        st.markdown(f"**Research Question:** {st.session_state.research_question}")
        
        st.markdown("---")
        st.markdown("### Resource Constraints")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            available_time = st.selectbox(
                "Available Time:",
                ["Short (< 6 months)", "Medium (6-12 months)", "Long (> 12 months)"],
                help="How much time do you have for data collection?"
            )
        
        with col_b:
            available_resources = st.selectbox(
                "Available Resources:",
                ["Limited", "Moderate", "Abundant"],
                help="What is your budget and resource availability?"
            )
        
        if st.button("Get Study Design Recommendations", type="primary"):
            recommendations = recommend_study_type(
                st.session_state.research_question,
                available_time,
                available_resources
            )
            
            st.markdown("---")
            st.markdown("### 🎯 Recommended Study Designs")
            
            for i, rec in enumerate(recommendations[:3], 1):
                with st.expander(f"**{i}. {rec['type']}** (Score: {rec['score']}/10)", expanded=(i==1)):
                    st.markdown(f"**Why this design?** {rec['reason']}")
                    
                    col_pros, col_cons = st.columns(2)
                    
                    with col_pros:
                        st.markdown("**✅ Advantages:**")
                        for pro in rec['pros']:
                            st.markdown(f"- {pro}")
                    
                    with col_cons:
                        st.markdown("**⚠️ Limitations:**")
                        for con in rec['cons']:
                            st.markdown(f"- {con}")
                    
                    if st.button(f"Select {rec['type']}", key=f"select_{i}"):
                        st.session_state.study_type = rec['type']
                        st.session_state.step = 3
                        st.rerun()
    
    with col2:
        st.markdown("### 📖 Study Design Guide")
        st.markdown("""
        **Cross-sectional:**
        - Snapshot in time
        - Prevalence studies
        
        **Case-Control:**
        - Retrospective
        - Good for rare diseases
        
        **Cohort:**
        - Prospective/Retrospective
        - Follow-up over time
        
        **RCT:**
        - Gold standard
        - Experimental
        
        **Diagnostic:**
        - Test evaluation
        - Sensitivity/Specificity
        """)
        
        if st.button("← Back to Research Topic"):
            st.session_state.step = 1
            st.rerun()

elif st.session_state.step == 3:
    # STEP 3: Sample Size Calculation
    st.markdown('<h2 class="sub-header">Step 3: Sample Size Calculation</h2>', unsafe_allow_html=True)
    
    st.markdown(f"**Research Topic:** {st.session_state.topic}")
    st.markdown(f"**Selected Study Design:** {st.session_state.study_type}")
    st.markdown("---")
    
    # Common parameters
    st.markdown("### 🔧 Statistical Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_level = st.selectbox(
            "Confidence Level:",
            [90, 95, 99],
            index=1,
            help="Probability that the true value lies within the confidence interval"
        )
    
    with col2:
        power = st.selectbox(
            "Statistical Power:",
            [80, 85, 90, 95],
            index=0,
            help="Probability of detecting a true effect"
        ) if st.session_state.study_type != "Cross-sectional Study" else None
    
    with col3:
        dropout_rate = st.slider(
            "Expected Dropout Rate (%):",
            0, 50, 10,
            help="Percentage of participants expected to drop out"
        )
    
    st.markdown("---")
    
    # Study-specific parameters
    if st.session_state.study_type == "Cross-sectional Study":
        st.markdown("### 📋 Cross-sectional Study Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            expected_prevalence = st.slider(
                "Expected Prevalence (%):",
                1, 99, 50,
                help="Expected proportion of the condition in the population"
            )
        
        with col2:
            margin_error = st.slider(
                "Margin of Error (%):",
                1, 20, 5,
                help="Acceptable difference from the true prevalence"
            )
        
        if st.button("Calculate Sample Size", type="primary"):
            n_adjusted, n_original = calculate_cross_sectional(
                expected_prevalence, confidence_level, margin_error, dropout_rate
            )
            
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"**Required Sample Size:** {n_adjusted}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.info(f"""
                **Breakdown:**
                - Base sample size: {n_original}
                - Adjusted for {dropout_rate}% dropout: {n_adjusted}
                """)
            
            with col2:
                st.markdown("### 📝 Interpretation")
                st.markdown(f"""
                To estimate the prevalence of your condition with:
                - **{confidence_level}% confidence**
                - **±{margin_error}% margin of error**
                - Expected prevalence of **{expected_prevalence}%**
                
                You need to recruit **{n_adjusted} participants**.
                """)
            
            # Additional information
            st.markdown("---")
            st.markdown("### 💡 Additional Considerations")
            st.markdown("""
            - Ensure random sampling to avoid selection bias
            - Consider stratification if studying multiple subgroups
            - Plan for slightly higher recruitment to account for incomplete data
            - Consider cluster sampling if individual sampling is not feasible
            """)
    
    elif st.session_state.study_type == "Case-Control Study":
        st.markdown("### 📋 Case-Control Study Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            odds_ratio = st.number_input(
                "Expected Odds Ratio:",
                min_value=1.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Expected strength of association"
            )
        
        with col2:
            control_exposure = st.slider(
                "Exposure in Controls (%):",
                1, 99, 20,
                help="Expected proportion of exposed individuals in control group"
            )
        
        with col3:
            case_control_ratio = st.selectbox(
                "Control:Case Ratio:",
                [1, 2, 3, 4],
                index=0,
                help="Number of controls per case"
            )
        
        if st.button("Calculate Sample Size", type="primary"):
            n_cases_adj, n_controls_adj, n_cases, n_controls = calculate_case_control(
                power, confidence_level, odds_ratio, control_exposure, case_control_ratio, dropout_rate
            )
            
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"**Cases needed:** {n_cases_adj}")
                st.markdown(f"**Controls needed:** {n_controls_adj}")
                st.markdown(f"**Total sample size:** {n_cases_adj + n_controls_adj}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.info(f"""
                **Breakdown:**
                - Base cases: {n_cases}
                - Base controls: {n_controls}
                - Adjusted for {dropout_rate}% dropout:
                  - Cases: {n_cases_adj}
                  - Controls: {n_controls_adj}
                """)
            
            with col2:
                st.markdown("### 📝 Interpretation")
                st.markdown(f"""
                To detect an odds ratio of **{odds_ratio}** with:
                - **{power}% power**
                - **{confidence_level}% confidence**
                - **{control_exposure}% exposure in controls**
                - **{case_control_ratio}:1 control-to-case ratio**
                
                You need:
                - **{n_cases_adj} cases**
                - **{n_controls_adj} controls**
                - **Total: {n_cases_adj + n_controls_adj} participants**
                """)
            
            st.markdown("---")
            st.markdown("### 💡 Additional Considerations")
            st.markdown("""
            - Match controls to cases on key confounders if needed
            - Consider using more controls per case for efficiency
            - Ensure clear case definition criteria
            - Minimize recall bias through objective exposure measurement
            - Consider prevalent vs. incident cases
            """)
    
    elif st.session_state.study_type == "Cohort Study":
        st.markdown("### 📋 Cohort Study Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            relative_risk = st.number_input(
                "Expected Relative Risk:",
                min_value=1.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Expected strength of association"
            )
        
        with col2:
            control_incidence = st.slider(
                "Incidence in Unexposed (%):",
                1, 50, 10,
                help="Expected incidence in unexposed group"
            )
        
        with col3:
            exposed_unexposed_ratio = st.selectbox(
                "Exposed:Unexposed Ratio:",
                [1, 2, 3, 4],
                index=0,
                help="Ratio of exposed to unexposed participants"
            )
        
        if st.button("Calculate Sample Size", type="primary"):
            result = calculate_cohort(
                power, confidence_level, relative_risk, control_incidence, 
                exposed_unexposed_ratio, dropout_rate
            )
            
            if result[0] is not None:
                n_exposed_adj, n_unexposed_adj, n_exposed, n_unexposed = result
                
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f"**Exposed needed:** {n_exposed_adj}")
                    st.markdown(f"**Unexposed needed:** {n_unexposed_adj}")
                    st.markdown(f"**Total sample size:** {n_exposed_adj + n_unexposed_adj}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.info(f"""
                    **Breakdown:**
                    - Base exposed: {n_exposed}
                    - Base unexposed: {n_unexposed}
                    - Adjusted for {dropout_rate}% dropout:
                      - Exposed: {n_exposed_adj}
                      - Unexposed: {n_unexposed_adj}
                    """)
                
                with col2:
                    st.markdown("### 📝 Interpretation")
                    st.markdown(f"""
                    To detect a relative risk of **{relative_risk}** with:
                    - **{power}% power**
                    - **{confidence_level}% confidence**
                    - **{control_incidence}% incidence in unexposed**
                    - **{exposed_unexposed_ratio}:1 exposed-to-unexposed ratio**
                    
                    You need:
                    - **{n_exposed_adj} exposed participants**
                    - **{n_unexposed_adj} unexposed participants**
                    - **Total: {n_exposed_adj + n_unexposed_adj} participants**
                    """)
                
                st.markdown("---")
                st.markdown("### 💡 Additional Considerations")
                st.markdown("""
                - Plan for long follow-up period
                - Implement strategies to minimize loss to follow-up
                - Consider interim analyses for long studies
                - Ensure consistent outcome assessment
                - Account for competing risks if relevant
                - Consider prospective vs. retrospective design trade-offs
                """)
    
    elif st.session_state.study_type == "Randomized Controlled Trial (RCT)":
        st.markdown("### 📋 RCT Parameters")
        
        outcome_type = st.radio(
            "Primary Outcome Type:",
            ["Continuous (e.g., blood pressure, weight)", "Binary (e.g., death, cure)"],
            help="Select the type of your primary outcome measure"
        )
        
        if outcome_type == "Continuous (e.g., blood pressure, weight)":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                control_mean = st.number_input(
                    "Control Group Mean:",
                    min_value=0.0,
                    value=100.0,
                    help="Expected mean in control group"
                )
            
            with col2:
                control_sd = st.number_input(
                    "Standard Deviation:",
                    min_value=0.1,
                    value=15.0,
                    help="Expected standard deviation"
                )
            
            with col3:
                effect_size = st.number_input(
                    "Expected Difference:",
                    min_value=0.1,
                    value=10.0,
                    help="Minimum clinically important difference"
                )
            
            with col4:
                allocation_ratio = st.selectbox(
                    "Allocation Ratio (I:C):",
                    [1, 2],
                    index=0,
                    help="Intervention to control ratio"
                )
            
            if st.button("Calculate Sample Size", type="primary"):
                n_intervention_adj, n_control_adj, n_intervention, n_control = calculate_rct(
                    power, confidence_level, effect_size, control_mean, control_sd, 
                    allocation_ratio, dropout_rate
                )
                
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f"**Intervention group:** {n_intervention_adj}")
                    st.markdown(f"**Control group:** {n_control_adj}")
                    st.markdown(f"**Total sample size:** {n_intervention_adj + n_control_adj}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.info(f"""
                    **Breakdown:**
                    - Base intervention: {n_intervention}
                    - Base control: {n_control}
                    - Adjusted for {dropout_rate}% dropout:
                      - Intervention: {n_intervention_adj}
                      - Control: {n_control_adj}
                    """)
                
                with col2:
                    st.markdown("### 📝 Interpretation")
                    st.markdown(f"""
                    To detect a difference of **{effect_size}** with:
                    - **{power}% power**
                    - **{confidence_level}% confidence**
                    - Control mean: **{control_mean}**
                    - Standard deviation: **{control_sd}**
                    - **{allocation_ratio}:1 allocation ratio**
                    
                    You need:
                    - **{n_intervention_adj} intervention participants**
                    - **{n_control_adj} control participants**
                    - **Total: {n_intervention_adj + n_control_adj} participants**
                    """)
        
        else:  # Binary outcome
            col1, col2, col3 = st.columns(3)
            
            with col1:
                control_event_rate = st.slider(
                    "Control Event Rate (%):",
                    1, 99, 30,
                    help="Expected event rate in control group"
                )
            
            with col2:
                intervention_event_rate = st.slider(
                    "Intervention Event Rate (%):",
                    1, 99, 15,
                    help="Expected event rate in intervention group"
                )
            
            with col3:
                allocation_ratio = st.selectbox(
                    "Allocation Ratio (I:C):",
                    [1, 2],
                    index=0,
                    help="Intervention to control ratio"
                )
            
            if st.button("Calculate Sample Size", type="primary"):
                n_intervention_adj, n_control_adj, n_intervention, n_control = calculate_rct_binary(
                    power, confidence_level, control_event_rate, intervention_event_rate,
                    allocation_ratio, dropout_rate
                )
                
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f"**Intervention group:** {n_intervention_adj}")
                    st.markdown(f"**Control group:** {n_control_adj}")
                    st.markdown(f"**Total sample size:** {n_intervention_adj + n_control_adj}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    absolute_risk_reduction = control_event_rate - intervention_event_rate
                    relative_risk_reduction = (absolute_risk_reduction / control_event_rate) * 100
                    
                    st.info(f"""
                    **Breakdown:**
                    - Base intervention: {n_intervention}
                    - Base control: {n_control}
                    - Adjusted for {dropout_rate}% dropout:
                      - Intervention: {n_intervention_adj}
                      - Control: {n_control_adj}
                    
                    **Effect Measures:**
                    - Absolute risk reduction: {absolute_risk_reduction:.1f}%
                    - Relative risk reduction: {relative_risk_reduction:.1f}%
                    - Number needed to treat: {100/absolute_risk_reduction:.1f}
                    """)
                
                with col2:
                    st.markdown("### 📝 Interpretation")
                    st.markdown(f"""
                    To detect a difference from **{control_event_rate}%** to **{intervention_event_rate}%** with:
                    - **{power}% power**
                    - **{confidence_level}% confidence**
                    - **{allocation_ratio}:1 allocation ratio**
                    
                    You need:
                    - **{n_intervention_adj} intervention participants**
                    - **{n_control_adj} control participants**
                    - **Total: {n_intervention_adj + n_control_adj} participants**
                    """)
        
        st.markdown("---")
        st.markdown("### 💡 Additional Considerations for RCTs")
        st.markdown("""
        - Use proper randomization methods (block, stratified)
        - Implement allocation concealment
        - Consider blinding (single, double, or triple)
        - Plan interim analyses carefully (may affect sample size)
        - Register trial before recruitment
        - Consider adaptive designs for efficiency
        - Account for non-adherence in analysis
        - Plan for intention-to-treat analysis
        """)
    
    elif st.session_state.study_type == "Diagnostic Accuracy Study":
        st.markdown("### 📋 Diagnostic Study Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sensitivity = st.slider(
                "Expected Sensitivity (%):",
                50, 99, 85,
                help="Expected true positive rate"
            )
        
        with col2:
            specificity = st.slider(
                "Expected Specificity (%):",
                50, 99, 90,
                help="Expected true negative rate"
            )
        
        with col3:
            prevalence = st.slider(
                "Disease Prevalence (%):",
                1, 99, 20,
                help="Expected prevalence of disease in study population"
            )
        
        margin_error = st.slider(
            "Margin of Error (%):",
            1, 20, 5,
            help="Acceptable difference in sensitivity/specificity estimation"
        )
        
        if st.button("Calculate Sample Size", type="primary"):
            n_total_adj, n_total, n_diseased, n_non_diseased = calculate_diagnostic(
                sensitivity, specificity, prevalence, confidence_level, margin_error, dropout_rate
            )
            
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"**Total sample size:** {n_total_adj}")
                st.markdown(f"**Diseased needed:** {ceil(n_total_adj * prevalence / 100)}")
                st.markdown(f"**Non-diseased needed:** {ceil(n_total_adj * (100 - prevalence) / 100)}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.info(f"""
                **Breakdown:**
                - Base total: {n_total}
                - For sensitivity: {n_diseased} diseased
                - For specificity: {n_non_diseased} non-diseased
                - Adjusted for {dropout_rate}% dropout: {n_total_adj}
                """)
            
            with col2:
                st.markdown("### 📝 Interpretation")
                st.markdown(f"""
                To estimate diagnostic accuracy with:
                - Expected sensitivity: **{sensitivity}%**
                - Expected specificity: **{specificity}%**
                - **{confidence_level}% confidence**
                - **±{margin_error}% margin of error**
                - Disease prevalence: **{prevalence}%**
                
                You need **{n_total_adj} total participants**, including:
                - **{ceil(n_total_adj * prevalence / 100)} with disease**
                - **{ceil(n_total_adj * (100 - prevalence) / 100)} without disease**
                """)
                
                # Calculate additional metrics
                ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (100 - specificity) * (100 - prevalence)) * 100
                npv = (specificity * (100 - prevalence)) / (specificity * (100 - prevalence) + (100 - sensitivity) * prevalence) * 100
                
                st.success(f"""
                **Expected Performance:**
                - Positive Predictive Value: {ppv:.1f}%
                - Negative Predictive Value: {npv:.1f}%
                """)
            
            st.markdown("---")
            st.markdown("### 💡 Additional Considerations")
            st.markdown("""
            - Use appropriate gold standard for comparison
            - Avoid spectrum bias (include full range of disease severity)
            - Blind test interpreters to diagnosis
            - Avoid verification bias (ensure all participants get gold standard)
            - Report STARD guidelines compliance
            - Consider ROC analysis for continuous test results
            - Account for indeterminate results
            """)
    
    # Back button
    st.markdown("---")
    if st.button("← Back to Study Design Recommendation"):
        st.session_state.step = 2
        st.rerun()
    
    # Export results button
    if st.button("📥 Export Results"):
        st.success("Results exported! (Feature to be implemented)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Medical Sample Size Calculator</strong></p>
    <p>Developed for medical students and researchers</p>
    <p>⚠️ <em>These calculations are estimates. Always consult with a statistician for your actual study.</em></p>
</div>

""", unsafe_allow_html=True)


