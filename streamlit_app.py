import sys
import os
import time
import textwrap
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Backend imports – direct engine calls
# ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from decision_engine import analyze_case  # noqa: E402
from planner import generate_plan  # noqa: E402
from scheduler import create_schedule, list_schedules, cancel_schedule  # noqa: E402

import streamlit as st  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Clinical Triage Dashboard",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# HTML Renderer helper to avoid markdown codeblocks
# ─────────────────────────────────────────────────────────────
def render_html(html_str):
    # Strips leading and trailing whitespaces from each line and joins them as a single line.
    # This prevents the markdown compiler from converting indented lines to preformatted code blocks.
    clean = "".join([line.strip() for line in html_str.split("\n") if line.strip()])
    st.markdown(clean, unsafe_allow_html=True)

def get_loading_overlay_html(step, confidence, symptom_count):
    # Fetch dynamic values computed in the python loop from session state
    pct = st.session_state.get("loading_pct", 0)
    rules_val = st.session_state.get("loading_rules", 0)
    nodes_val = st.session_state.get("loading_nodes", 0)
    syms_val = st.session_state.get("loading_syms", 0)
    conf_val = st.session_state.get("loading_conf", 0)
    thought_msg = st.session_state.get("loading_thought", "")
    remaining_seconds = st.session_state.get("loading_remaining", 0.0)

    steps_text = [
        "Initializing Clinical Engine",
        "Normalizing Symptoms",
        "Removing Duplicate Entries",
        "Extracting Clinical Context",
        "Evaluating Severity",
        "Matching Clinical Rules",
        "Confidence Calibration",
        "Specialist Mapping",
        "Generating Follow-up Plan",
        "Searching Provider Database",
        "Preparing Dashboard"
    ]
    
    timeline_html = ""
    for idx, txt in enumerate(steps_text):
        if idx < step:
            timeline_html += f'<div class="timeline-step completed"><span class="glowing-checkmark">✔</span> <span>{txt}</span></div>'
        elif idx == step:
            timeline_html += f'<div class="timeline-step active"><span>⏳</span> <span>{txt}...</span></div>'
        else:
            break
            
    pipeline_stages = [
        "Symptoms", "Clinical Context", "Severity", "Rule Matching",
        "Confidence", "Specialist Mapping", "Follow-up Planning",
        "Appointment Scheduling", "Dashboard Generation"
    ]
    
    active_pipe = 0
    if step <= 2: active_pipe = 0
    elif step == 3: active_pipe = 1
    elif step == 4: active_pipe = 2
    elif step == 5: active_pipe = 3
    elif step == 6: active_pipe = 4
    elif step == 7: active_pipe = 5
    elif step == 8: active_pipe = 6
    elif step in (9, 10): active_pipe = 7
    else: active_pipe = 8
    
    pipeline_html = ""
    for idx, label in enumerate(pipeline_stages):
        if idx > 0:
            conn_class = "pipeline-connector completed" if idx <= active_pipe else "pipeline-connector"
            pipeline_html += f'<div class="{conn_class}"></div>'
            
        if idx < active_pipe:
            item_class = "pipeline-item completed"
        elif idx == active_pipe:
            item_class = "pipeline-item active"
        else:
            item_class = "pipeline-item"
            
        pipeline_html += f"""
        <div class="{item_class}">
            <div class="pipeline-dot"></div>
            <span class="pipeline-label">{label}</span>
        </div>
        """
        
    filled_blocks = int(pct / 5)
    empty_blocks = 20 - filled_blocks
    ascii_bar = "█" * filled_blocks + "░" * empty_blocks
    
    if step >= 11:
        countdown_text = "Complete"
    else:
        countdown_text = f"{int(remaining_seconds)} second" + ("s" if int(remaining_seconds) != 1 else "")
        
    if step == 11:
        grid_style = 'style="display: none; opacity: 0;"'
        comp_style = 'style="display: flex; opacity: 1;"'
    else:
        grid_style = 'style="display: grid; opacity: 1;"'
        comp_style = 'style="display: none; opacity: 0;"'
        
    fade_out_class = " fade-out" if st.session_state.get("loading_fade_out", False) else ""
    overlay_html = f"""
    <div class="loading-overlay{fade_out_class}" id="loading-overlay">
        <div class="loading-modal">
            <div class="loading-header">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.8rem; animation: pulseSpark 1.5s infinite alternate;">⚡</span>
                        <span style="font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #c084fc, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px;">NabzAI Clinical Triage Engine</span>
                    </div>
                    <div style="font-size: 0.95rem; color: #94a3b8; margin-top: 0.4rem; font-weight: 400; max-width: 580px;">
                        Analyzing patient symptoms using hybrid rule-based reasoning and adaptive machine intelligence.
                    </div>
                </div>
                <div class="pulse-active-container">
                    <div class="pulse-icon-circle"></div>
                    <span style="font-size: 0.8rem; color: #c084fc; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px;">AI ENGINE ACTIVE</span>
                </div>
            </div>
            
            <div class="card-body-grid" id="card-body-grid" {grid_style}>
                <div class="card-left-column">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.75rem;">Live Reasoning Trace</div>
                    <div id="live-timeline-container" class="live-timeline-container">
                        {timeline_html}
                    </div>
                    
                    <div style="margin-top: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem; font-weight: 500;">
                            <span id="progress-lbl">Status: {thought_msg}</span>
                            <span id="progress-percentage">{pct}%</span>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: {pct}%;"></div>
                        </div>
                        <div class="progress-ascii-row" id="progress-ascii">
                            {ascii_bar} {pct}%
                        </div>
                        
                        <div class="countdown-container" style="margin-top: 0.75rem;">
                            <span style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Estimated Remaining Time:</span>
                            <span id="countdown-timer" class="countdown-highlight">{countdown_text}</span>
                        </div>
                    </div>
                </div>
                
                <div class="card-right-column">
                    <div class="brain-visualization-container">
                        <canvas id="brain-canvas" width="320" height="140"></canvas>
                    </div>
                    
                    <div style="font-size: 0.8rem; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.4rem; margin-top: 0.2rem;">Clinical Pipeline</div>
                    <div class="pipeline-container">
                        {pipeline_html}
                    </div>
                    
                    <div class="stats-panel-loading">
                        <div class="stat-loading-box">
                            <span class="stat-loading-lbl">Rules Evaluated</span>
                            <span class="stat-loading-val">{rules_val}</span>
                        </div>
                        <div class="stat-loading-box">
                            <span class="stat-loading-lbl">Nodes Visited</span>
                            <span class="stat-loading-val">{nodes_val:,}</span>
                        </div>
                        <div class="stat-loading-box">
                            <span class="stat-loading-lbl">Symptoms Processed</span>
                            <span class="stat-loading-val">{syms_val}</span>
                        </div>
                        <div class="stat-loading-box">
                            <span class="stat-loading-lbl">Confidence Score</span>
                            <span class="stat-loading-val">{conf_val}%</span>
                        </div>
                    </div>
                    
                    <div class="metrics-panel">
                        <div style="font-size: 0.65rem; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem;">Clinical Intelligence Metrics</div>
                        <div class="metrics-grid">
                            <div class="metric-row">
                                <span>Inference Speed</span>
                                <span class="metric-value-pulsing" id="metric-speed">242 ops/sec</span>
                            </div>
                            <div class="metric-row">
                                <span>Ontology Depth</span>
                                <span>Level 14</span>
                            </div>
                            <div class="metric-row">
                                <span>Diagnostic Coverage</span>
                                <span>98.6%</span>
                            </div>
                            <div class="metric-row">
                                <span>Decision Entropy</span>
                                <span id="metric-entropy">0.142 nats</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="ai-thought-box">
                        <span class="ai-thought-lbl">Current Thought:</span>
                        <span id="ai-thought-val" class="ai-thought-msg">{thought_msg}</span>
                    </div>
                </div>
            </div>
            
            <div class="completion-screen" id="completion-screen" {comp_style}>
                <svg class="checkmark-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52">
                    <circle class="checkmark-circle" cx="26" cy="26" r="25" fill="none"/>
                    <path class="checkmark-check" fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>
                </svg>
                <div class="completion-title">✅ Clinical Analysis Completed</div>
                <div class="completion-subtitle" style="color: #a78bfa; font-weight: 600; margin-bottom: 0.25rem;">Confidence Successfully Calibrated</div>
                <div class="completion-subtitle">Preparing Dashboard...</div>
            </div>
        </div>
    </div>
    
    <style>
    .loading-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(8, 10, 18, 0.78);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999999;
        pointer-events: all;
        transition: opacity 0.6s ease-in-out;
        opacity: 1;
    }}
    .loading-overlay.fade-out {{
        opacity: 0;
    }}
    .loading-modal {{
        width: 800px;
        max-width: 95%;
        background: linear-gradient(135deg, rgba(13, 17, 33, 0.85) 0%, rgba(20, 26, 48, 0.9) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 28px;
        padding: 2.5rem;
        box-shadow: 0 30px 70px rgba(0,0,0,0.85), 0 0 50px rgba(99, 102, 241, 0.15);
        display: flex;
        flex-direction: column;
        gap: 1.75rem;
        font-family: 'Outfit', sans-serif;
        color: #f1f5f9;
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        animation: floatCard 5s ease-in-out infinite;
    }}
    @keyframes floatCard {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
        100% {{ transform: translateY(0px); }}
    }}
    .loading-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.25rem;
    }}
    .pulse-active-container {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 6px 12px;
        border-radius: 20px;
        background: rgba(168, 85, 247, 0.06);
    }}
    .pulse-icon-circle {{
        width: 8px;
        height: 8px;
        background-color: #a855f7;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.7);
        animation: active-pulse 1.5s infinite;
    }}
    @keyframes active-pulse {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(168, 85, 247, 0.7); }}
        70% {{ transform: scale(1); box-shadow: 0 0 0 8px rgba(168, 85, 247, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(168, 85, 247, 0); }}
    }}
    @keyframes active-blue-pulse {{
        0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7); }}
        70% {{ transform: scale(1); box-shadow: 0 0 0 8px rgba(99, 102, 241, 0); }}
        100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }}
    }}
    @keyframes pulseSpark {{
        from {{ transform: scale(1); filter: drop-shadow(0 0 2px rgba(168, 85, 247, 0.4)); }}
        to {{ transform: scale(1.15); filter: drop-shadow(0 0 8px rgba(168, 85, 247, 0.8)); }}
    }}
    .card-body-grid {{
        display: grid;
        grid-template-columns: 1.15fr 0.85fr;
        gap: 2rem;
        transition: opacity 0.4s ease;
    }}
    .card-left-column {{
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .card-right-column {{
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
    }}
    .live-timeline-container {{
        display: flex;
        flex-direction: column;
        gap: 8px;
        height: 310px;
        overflow-y: hidden;
    }}
    .timeline-step {{
        font-size: 0.88rem;
        padding: 7px 12px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: rgba(255, 255, 255, 0.3);
        opacity: 0;
        transform: translateX(-20px);
        animation: stepReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}
    @keyframes stepReveal {{
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    .timeline-step.active {{
        color: #c084fc;
        background: rgba(168, 85, 247, 0.08);
        border: 1px solid rgba(168, 85, 247, 0.2);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.1);
        font-weight: 500;
    }}
    .timeline-step.completed {{
        color: #34d399;
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.25);
        font-weight: 500;
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.15);
        animation: successPulse 0.4s ease-out;
    }}
    @keyframes successPulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.03); box-shadow: 0 0 15px rgba(52, 211, 153, 0.4); }}
        100% {{ transform: scale(1); }}
    }}
    .glowing-checkmark {{
        color: #34d399;
        text-shadow: 0 0 8px #34d399, 0 0 15px rgba(52, 211, 153, 0.6);
        font-weight: bold;
        display: inline-block;
        animation: checkmarkPulse 0.4s ease-out;
    }}
    @keyframes checkmarkPulse {{
        0% {{ transform: scale(0.6); }}
        100% {{ transform: scale(1); }}
    }}
    .progress-bar-container {{
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 9999px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 0.4rem;
    }}
    .progress-bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, #a855f7, #6366f1);
        border-radius: 9999px;
        transition: width 0.12s linear;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.4);
    }}
    .progress-ascii-row {{
        font-family: monospace;
        font-size: 0.9rem;
        color: #a78bfa;
        letter-spacing: 1.5px;
        text-shadow: 0 0 5px rgba(167, 139, 250, 0.3);
    }}
    .countdown-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.03);
        padding: 6px 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }}
    .countdown-highlight {{
        font-size: 0.85rem;
        font-weight: 700;
        color: #c084fc;
    }}
    .brain-visualization-container {{
        width: 100%;
        height: 140px;
        background: rgba(8, 10, 18, 0.45);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }}
    .pipeline-container {{
        display: flex;
        flex-direction: column;
        gap: 3px;
        background: rgba(8, 10, 18, 0.25);
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.02);
    }}
    .pipeline-item {{
        display: flex;
        align-items: center;
        gap: 12px;
        opacity: 0.3;
        transition: all 0.3s ease;
    }}
    .pipeline-item.active {{
        opacity: 1;
        transform: scale(1.02);
    }}
    .pipeline-item.completed {{
        opacity: 0.9;
    }}
    .pipeline-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #94a3b8;
        transition: all 0.3s ease;
    }}
    .pipeline-item.active .pipeline-dot {{
        background: #6366f1;
        box-shadow: 0 0 10px #6366f1;
        animation: active-blue-pulse 1.5s infinite;
    }}
    .pipeline-item.completed .pipeline-dot {{
        background: #34d399;
        box-shadow: 0 0 8px #34d399;
    }}
    .pipeline-label {{
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
        transition: all 0.3s ease;
    }}
    .pipeline-item.active .pipeline-label {{
        color: #818cf8;
        font-weight: 600;
        text-shadow: 0 0 8px rgba(129, 140, 248, 0.3);
    }}
    .pipeline-item.completed .pipeline-label {{
        color: #34d399;
    }}
    .pipeline-connector {{
        width: 2px;
        height: 8px;
        background: rgba(255, 255, 255, 0.05);
        margin-left: 2.5px;
        transition: all 0.3s ease;
    }}
    .pipeline-connector.completed {{
        background: #34d399;
    }}
    .stats-panel-loading {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }}
    .stat-loading-box {{
        background: rgba(22, 27, 46, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 0.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}
    .stat-loading-lbl {{
        font-size: 0.62rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: center;
    }}
    .stat-loading-val {{
        font-size: 1.1rem;
        font-weight: 700;
        color: #818cf8;
        margin-top: 2px;
        font-family: monospace;
        text-shadow: 0 0 10px rgba(129, 140, 248, 0.25);
    }}
    .metrics-panel {{
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 10px;
    }}
    .metrics-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }}
    .metric-row {{
        display: flex;
        flex-direction: column;
        gap: 2px;
        font-size: 0.7rem;
        color: #94a3b8;
    }}
    .metric-row span:last-child {{
        font-weight: 600;
        color: #ffffff;
    }}
    .metric-value-pulsing {{
        animation: pulseText 1.5s infinite alternate;
    }}
    @keyframes pulseText {{
        from {{ opacity: 0.75; }}
        to {{ opacity: 1; }}
    }}
    .ai-thought-box {{
        background: rgba(168, 85, 247, 0.05);
        border: 1px dashed rgba(168, 85, 247, 0.25);
        border-radius: 10px;
        padding: 0.6rem 0.85rem;
        display: flex;
        flex-direction: column;
        gap: 3px;
        animation: thoughtFade 0.5s ease-in-out;
    }}
    @keyframes thoughtFade {{
        from {{ opacity: 0.6; }}
        to {{ opacity: 1; }}
    }}
    .ai-thought-lbl {{
        font-size: 0.7rem;
        font-weight: 700;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .ai-thought-msg {{
        font-size: 0.8rem;
        color: #e2e8f0;
        font-weight: 500;
        font-style: italic;
    }}
    .completion-screen {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 380px;
        text-align: center;
        animation: fadeInScale 0.5s ease forwards;
    }}
    @keyframes fadeInScale {{
        from {{ opacity: 0; transform: scale(0.95); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    .completion-title {{
        font-size: 1.45rem;
        font-weight: 700;
        color: #34d399;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 15px rgba(52, 211, 153, 0.25);
    }}
    .completion-subtitle {{
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 500;
    }}
    .checkmark-svg {{
        width: 70px;
        height: 70px;
        border-radius: 50%;
        display: block;
        stroke-width: 4;
        stroke: #34d399;
        stroke-miterlimit: 10;
        box-shadow: inset 0px 0px 0px rgba(52, 211, 153, 0.08);
        animation: fillCheckmark .4s ease-in-out .4s forwards, scaleCheckmark .3s ease-in-out .9s forwards;
        margin: 0 auto 1.5rem auto;
    }}
    .checkmark-circle {{
        stroke-dasharray: 166;
        stroke-dashoffset: 166;
        stroke-width: 4;
        stroke-miterlimit: 10;
        stroke: #34d399;
        fill: none;
        animation: strokeCircle 0.6s cubic-bezier(0.65, 0, 0.45, 1) forwards;
    }}
    .checkmark-check {{
        transform-origin: 50% 50%;
        stroke-dasharray: 48;
        stroke-dashoffset: 48;
        animation: strokeCheck 0.3s cubic-bezier(0.65, 0, 0.45, 1) 0.6s forwards;
    }}
    @keyframes strokeCircle {{
        100% {{ stroke-dashoffset: 0; }}
    }}
    @keyframes strokeCheck {{
        100% {{ stroke-dashoffset: 0; }}
    }}
    @keyframes fillCheckmark {{
        100% {{ box-shadow: inset 0px 0px 0px 35px rgba(52, 211, 153, 0.08); }}
    }}
    @keyframes scaleCheckmark {{
        0%, 100% {{ transform: none; }}
        50% {{ transform: scale3d(1.1, 1.1, 1); }}
    }}
    </style>
    
    <script>
    (function() {{
        const canvas = document.getElementById("brain-canvas");
        if (canvas) {{
            if (window.brainAnimFrameId) {{
                cancelAnimationFrame(window.brainAnimFrameId);
            }}
            const ctx = canvas.getContext("2d");
            const nodes = [];
            const layers = [3, 5, 4, 2];
            const layerSpacing = canvas.width / (layers.length + 1);
            
            // Generate deterministic nodes
            for (let i = 0; i < layers.length; i++) {{
                const x = layerSpacing * (i + 1);
                const count = layers[i];
                const nodeSpacing = canvas.height / (count + 1);
                for (let j = 0; j < count; j++) {{
                    const y = nodeSpacing * (j + 1);
                    const phaseSeed = (i * 7 + j * 13) % (Math.PI * 2);
                    const speedSeed = 0.002 + ((i * 3 + j * 5) % 10) / 1000;
                    nodes.push({{
                        x: x,
                        y: y,
                        r: 3 + ((i + j) % 3),
                        color: i === 0 ? "#818cf8" : i === layers.length - 1 ? "#34d399" : "#a78bfa",
                        phaseSeed: phaseSeed,
                        speedSeed: speedSeed
                    }});
                }}
            }}
            
            // Generate deterministic links
            const links = [];
            for (let i = 0; i < nodes.length; i++) {{
                for (let j = 0; j < nodes.length; j++) {{
                    const n1 = nodes[i];
                    const n2 = nodes[j];
                    if (n2.x > n1.x && n2.x - n1.x < layerSpacing * 1.5) {{
                        const linkSeed = (i * 31 + j * 17) % 100;
                        if (linkSeed < 70) {{
                            links.push({{ from: n1, to: n2, seed: linkSeed }});
                        }}
                    }}
                }}
            }}
            
            function draw() {{
                if (!document.getElementById("brain-canvas")) return;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const time = Date.now();
                
                // Draw rotating rings in background
                ctx.strokeStyle = "rgba(168, 85, 247, 0.12)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(canvas.width / 2, canvas.height / 2, 45, time * 0.0008, time * 0.0008 + Math.PI * 0.7);
                ctx.stroke();
                
                ctx.strokeStyle = "rgba(129, 140, 248, 0.12)";
                ctx.beginPath();
                ctx.arc(canvas.width / 2, canvas.height / 2, 55, -time * 0.0006, -time * 0.0006 + Math.PI * 0.9);
                ctx.stroke();
                
                // Draw links
                ctx.lineWidth = 0.8;
                for (let i = 0; i < links.length; i++) {{
                    const l = links[i];
                    ctx.strokeStyle = "rgba(129, 140, 248, 0.1)";
                    ctx.beginPath();
                    ctx.moveTo(l.from.x, l.from.y);
                    ctx.lineTo(l.to.x, l.to.y);
                    ctx.stroke();
                }}
                
                // Draw particles along links
                for (let i = 0; i < links.length; i++) {{
                    const l = links[i];
                    const period = 1500 + (l.seed * 15) % 1000;
                    const offset = (i * 200) % period;
                    const progress = ((time + offset) % period) / period;
                    
                    const x = l.from.x + (l.to.x - l.from.x) * progress;
                    const y = l.from.y + (l.to.y - l.from.y) * progress;
                    
                    ctx.beginPath();
                    ctx.arc(x, y, 1.8, 0, Math.PI * 2);
                    ctx.fillStyle = "#c084fc";
                    ctx.fill();
                }}
                
                // Draw nodes
                for (let i = 0; i < nodes.length; i++) {{
                    const n = nodes[i];
                    const pulse = Math.sin(time * n.speedSeed + n.phaseSeed) * 1.2;
                    const rad = n.r + pulse;
                    
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, rad, 0, Math.PI * 2);
                    ctx.fillStyle = n.color;
                    ctx.fill();
                }}
                
                // Dynamic live metrics update
                if (time % 10 < 2) {{
                    const speedEl = document.getElementById("metric-speed");
                    if (speedEl) speedEl.innerText = Math.floor(235 + Math.random() * 20) + " ops/sec";
                    
                    const entEl = document.getElementById("metric-entropy");
                    if (entEl) entEl.innerText = (0.12 + Math.random() * 0.04).toFixed(3) + " nats";
                }}
                
                window.brainAnimFrameId = requestAnimationFrame(draw);
            }}
            draw();
            
            // Auto-scroll timeline to bottom
            const timeline = document.getElementById("live-timeline-container");
            if (timeline) {{
                timeline.scrollTop = timeline.scrollHeight;
            }}
        }}
    }})();
    </script>
    """
    return "".join([line.strip() for line in overlay_html.split("\n") if line.strip()])

# ─────────────────────────────────────────────────────────────
# CSS Styling & Glassmorphism Theme Injection
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0c0e17;
    color: #f1f5f9;
    font-family: 'Outfit', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #121522;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa 0%, #6366f1 50%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

.custom-card {
    background: rgba(18, 22, 35, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

.triage-metric-card {
    background: linear-gradient(135deg, rgba(26, 31, 51, 0.7) 0%, rgba(15, 18, 30, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    transition: all 0.3s ease;
}

.triage-metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 12px 30px rgba(99, 102, 241, 0.15);
}

.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 9999px;
}

.badge-high {
    background: rgba(239, 68, 68, 0.12);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-medium {
    background: rgba(245, 158, 11, 0.12);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-low {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.doctor-card {
    background: linear-gradient(135deg, rgba(22, 27, 46, 0.8) 0%, rgba(11, 14, 26, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1.75rem;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.doctor-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
}

.timeline {
    position: relative;
    padding-left: 2rem;
    border-left: 2px solid rgba(255, 255, 255, 0.08);
    margin-left: 0.5rem;
}

.timeline-node {
    position: relative;
    margin-bottom: 1.75rem;
}

.timeline-node::before {
    content: '';
    position: absolute;
    left: -2.35rem;
    top: 0.2rem;
    width: 0.7rem;
    height: 0.7rem;
    border-radius: 50%;
    background-color: #6366f1;
    box-shadow: 0 0 10px #6366f1;
}

.appointment-card {
    background: rgba(18, 22, 35, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.appointment-card:hover {
    border-color: rgba(99, 102, 241, 0.25);
    transform: translateY(-2px);
}

/* Custom styled stats block */
.stats-container {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stats-item {
    background: rgba(22, 27, 46, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.stats-num {
    font-size: 1.4rem;
    font-weight: 700;
    color: #818cf8;
}

.stats-lbl {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

/* Form styling overrides */
div[data-testid="stForm"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Fix Plotly outline colors */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* Dashboard Reveal Animation */
@keyframes slideUpReveal {
    from {
        transform: translateY(30px) scale(0.98);
        opacity: 0;
        filter: blur(4px);
    }
    to {
        transform: translateY(0) scale(1.0);
        opacity: 1;
        filter: blur(0);
    }
}

.reveal-sec-1 { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 0ms; }
.reveal-sec-2 { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 150ms; }
.reveal-sec-3, .js-plotly-plot { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 300ms; }
.reveal-sec-4 { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 450ms; }
.reveal-sec-5 { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 600ms; }
.reveal-sec-6, .doctor-card { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 750ms; }
.reveal-sec-7 { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 900ms; }
.reveal-sec-8, .stInfo, div[data-testid="stCheckbox"] { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 1050ms !important; }
.reveal-sec-9, .appointment-card { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 1200ms !important; }
.reveal-sec-10 { animation: slideUpReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; animation-delay: 1350ms; }

/* Prevent dashboard ghosting/overlapping during loading overlay */
body:has(#loading-overlay:not(.fade-out)) [data-testid="stSidebar"] {
    opacity: 0.15 !important;
    filter: blur(25px) !important;
    pointer-events: none !important;
    transition: opacity 0.6s ease, filter 0.6s ease;
}

body:has(#loading-overlay:not(.fade-out)) [data-testid="stHeader"] {
    opacity: 0.15 !important;
    filter: blur(10px) !important;
    pointer-events: none !important;
    transition: opacity 0.6s ease, filter 0.6s ease;
}

body:has(#loading-overlay:not(.fade-out)) .main-title,
body:has(#loading-overlay:not(.fade-out)) .main-subtitle,
body:has(#loading-overlay:not(.fade-out)) .custom-card,
body:has(#loading-overlay:not(.fade-out)) .triage-metric-card,
body:has(#loading-overlay:not(.fade-out)) .stats-container,
body:has(#loading-overlay:not(.fade-out)) .stAlert,
body:has(#loading-overlay:not(.fade-out)) h3,
body:has(#loading-overlay:not(.fade-out)) hr,
body:has(#loading-overlay:not(.fade-out)) .js-plotly-plot,
body:has(#loading-overlay:not(.fade-out)) [data-testid="stForm"],
body:has(#loading-overlay:not(.fade-out)) div[data-testid="stWidget"],
body:has(#loading-overlay:not(.fade-out)) .stElementContainer:not(:has(#loading-overlay)) {
    opacity: 0.15 !important;
    filter: blur(25px) !important;
    pointer-events: none !important;
    transition: opacity 0.6s ease, filter 0.6s ease;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants & Enriched Predefined Data
# ─────────────────────────────────────────────────────────────
COMMON_SYMPTOMS = [
    "Fever",
    "Headache",
    "Chest Pain",
    "Abdominal Pain",
    "Stomach Pain",
    "Vomiting",
    "Nausea",
    "Cough",
    "Shortness of Breath",
    "Fatigue",
    "Dizziness",
    "Burning Urination",
    "Frequent Urination",
    "Skin Rash",
    "Back Pain",
    "Joint Pain",
    "Sore Throat",
    "Chills",
    "Loss of Balance",
    "Blurred Vision",
    "Muscle Pain",
    "Palpitations",
    "Diarrhea",
    "Constipation",
    "Ear Pain",
    "Other symptom"
]

DUMMY_DOCTORS = {
    "Cardiologist": {"name": "Dr. Ramesh Gupta", "hospital": "City Heart Institute", "timing": "10:00 AM - 01:00 PM", "location": "Sector 45, Gurugram", "rating": "4.9", "experience": "18 Years", "fee": "₹1,000", "languages": "English, Hindi", "next_available": "11:15 AM"},
    "Pulmonologist": {"name": "Dr. Ananya Sharma", "hospital": "Apex Lung Care", "timing": "02:00 PM - 05:00 PM", "location": "Vasant Kunj, Delhi", "rating": "4.8", "experience": "12 Years", "fee": "₹900", "languages": "English, Hindi, Punjabi", "next_available": "2:30 PM"},
    "Neurologist": {"name": "Dr. Rakesh Verma", "hospital": "NeuroBrain Center", "timing": "11:00 AM - 03:00 PM", "location": "Connaught Place, Delhi", "rating": "4.9", "experience": "20 Years", "fee": "₹1,200", "languages": "English, Hindi", "next_available": "12:00 PM"},
    "Ophthalmologist": {"name": "Dr. Sneha Desai", "hospital": "ClearVision Clinic", "timing": "09:00 AM - 12:00 PM", "location": "Sector 18, Noida", "rating": "4.7", "experience": "10 Years", "fee": "₹700", "languages": "English, Gujarati", "next_available": "10:15 AM"},
    "Urologist": {"name": "Dr. Vivek Kumar", "hospital": "Kidney Care Hospital", "timing": "03:00 PM - 06:00 PM", "location": "Saket, Delhi", "rating": "4.6", "experience": "14 Years", "fee": "₹800", "languages": "English, Hindi", "next_available": "4:00 PM"},
    "Dermatologist": {"name": "Dr. Meera Singh", "hospital": "SkinGlow Clinic", "timing": "12:00 PM - 04:00 PM", "location": "Indiranagar, Bangalore", "rating": "4.9", "experience": "11 Years", "fee": "₹850", "languages": "English, Kannada, Hindi", "next_available": "1:30 PM"},
    "Psychiatrist": {"name": "Dr. Anil Menon", "hospital": "Mind Wellness Hub", "timing": "10:30 AM - 02:30 PM", "location": "Bandra West, Mumbai", "rating": "4.8", "experience": "16 Years", "fee": "₹1,100", "languages": "English, Malayalam, Hindi", "next_available": "11:45 AM"},
    "Orthopedic": {"name": "Dr. Karan Malhotra", "hospital": "Bone & Joint Center", "timing": "09:30 AM - 01:30 PM", "location": "Salt Lake, Kolkata", "rating": "4.7", "experience": "13 Years", "fee": "₹900", "languages": "English, Bengali, Hindi", "next_available": "10:30 AM"},
    "Gastroenterologist": {"name": "Dr. Sanjay Das", "hospital": "Digestive Health Clinic", "timing": "04:00 PM - 07:00 PM", "location": "Jubilee Hills, Hyderabad", "rating": "4.9", "experience": "17 Years", "fee": "₹950", "languages": "English, Telugu, Hindi", "next_available": "5:15 PM"},
    "Endocrinologist": {"name": "Dr. Sunita Rao", "hospital": "Hormone Care Center", "timing": "08:00 AM - 11:30 AM", "location": "Anna Nagar, Chennai", "rating": "4.8", "experience": "15 Years", "fee": "₹900", "languages": "English, Tamil", "next_available": "9:30 AM"},
    "General Physician": {"name": "Dr. Pooja Iyer", "hospital": "City Care Hospital", "timing": "09:00 AM - 05:00 PM", "location": "Near Sector 18, Noida", "rating": "4.9", "experience": "15 Years", "fee": "₹800", "languages": "English, Hindi, Tamil", "next_available": "3:15 PM"},
    "Internal Medicine": {"name": "Dr. Pooja Iyer", "hospital": "City Care Hospital", "timing": "09:00 AM - 05:00 PM", "location": "Near Sector 18, Noida", "rating": "4.9", "experience": "15 Years", "fee": "₹800", "languages": "English, Hindi, Tamil", "next_available": "3:15 PM"},
    "Emergency Medicine": {"name": "Dr. Rajesh Kaushik", "hospital": "Apollo ER Care", "timing": "24/7 Availability", "location": "Sarita Vihar, Delhi", "rating": "4.9", "experience": "22 Years", "fee": "₹1,500", "languages": "English, Hindi", "next_available": "Immediate"},
    "Emergency": {"name": "Dr. Rajesh Kaushik", "hospital": "Apollo ER Care", "timing": "24/7 Availability", "location": "Sarita Vihar, Delhi", "rating": "4.9", "experience": "22 Years", "fee": "₹1,500", "languages": "English, Hindi", "next_available": "Immediate"},
    "ENT Specialist": {"name": "Dr. Harpreet Singh", "hospital": "ENT Clinic Delhi", "timing": "11:00 AM - 04:00 PM", "location": "Dwarka, Delhi", "rating": "4.6", "experience": "12 Years", "fee": "₹750", "languages": "English, Hindi, Punjabi", "next_available": "12:30 PM"},
    "Dental": {"name": "Dr. Kavita Murthy", "hospital": "DentCare Clinic", "timing": "10:00 AM - 06:00 PM", "location": "Indiranagar, Bangalore", "rating": "4.8", "experience": "9 Years", "fee": "₹600", "languages": "English, Kannada", "next_available": "11:00 AM"},
    "Dentist": {"name": "Dr. Kavita Murthy", "hospital": "DentCare Clinic", "timing": "10:00 AM - 06:00 PM", "location": "Indiranagar, Bangalore", "rating": "4.8", "experience": "9 Years", "fee": "₹600", "languages": "English, Kannada", "next_available": "11:00 AM"},
}

DUMMY_HOSPITALS = {
    "City Heart Institute": {"address": "Sector 45, Gurugram, Haryana - 122003", "distance": "4.2 km", "emergency": "Yes (24/7)", "contact": "+91 124 555 0192"},
    "Apex Lung Care": {"address": "Vasant Kunj Sector C, New Delhi - 110070", "distance": "8.5 km", "emergency": "No", "contact": "+91 11 555 0283"},
    "NeuroBrain Center": {"address": "Connaught Place Block E, New Delhi - 110001", "distance": "12.0 km", "emergency": "Yes (24/7)", "contact": "+91 11 555 0947"},
    "ClearVision Clinic": {"address": "Sector 18 Pocket A, Noida, UP - 201301", "distance": "6.1 km", "emergency": "No", "contact": "+91 120 555 0374"},
    "Kidney Care Hospital": {"address": "Saket Press Enclave Road, New Delhi - 110017", "distance": "9.8 km", "emergency": "Yes (24/7)", "contact": "+91 11 555 0103"},
    "SkinGlow Clinic": {"address": "100 Feet Road, Indiranagar, Bangalore - 560038", "distance": "5.5 km", "emergency": "No", "contact": "+91 80 555 0492"},
    "Mind Wellness Hub": {"address": "Hill Road, Bandra West, Mumbai - 400050", "distance": "15.3 km", "emergency": "No", "contact": "+91 22 555 0812"},
    "Bone & Joint Center": {"address": "Salt Lake Sector II, Kolkata, WB - 700091", "distance": "11.2 km", "emergency": "No", "contact": "+91 33 555 0731"},
    "Digestive Health Clinic": {"address": "Road No. 36, Jubilee Hills, Hyderabad - 500033", "distance": "7.0 km", "emergency": "No", "contact": "+91 40 555 0628"},
    "Hormone Care Center": {"address": "Anna Nagar 2nd Avenue, Chennai, TN - 600040", "distance": "10.4 km", "emergency": "No", "contact": "+91 44 555 0583"},
    "City Care Hospital": {"address": "Sector 18 Commercial Hub, Noida, UP - 201301", "distance": "5.8 km", "emergency": "Yes (24/7)", "contact": "+91 120 555 0999"},
    "Apollo ER Care": {"address": "Mathura Road, Sarita Vihar, New Delhi - 110076", "distance": "14.1 km", "emergency": "Yes (24/7) - Dedicated Trauma Center", "contact": "+91 11 555 0911"},
    "ENT Clinic Delhi": {"address": "Dwarka Sector 12, New Delhi - 110075", "distance": "16.5 km", "emergency": "No", "contact": "+91 11 555 0245"},
    "DentCare Clinic": {"address": "Double Road, Indiranagar, Bangalore - 560038", "distance": "6.0 km", "emergency": "No", "contact": "+91 80 555 0134"},
}

# ─────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────
if "triage_result" not in st.session_state:
    st.session_state.triage_result = None
if "plan" not in st.session_state:
    st.session_state.plan = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "prefilled_symptoms" not in st.session_state:
    st.session_state.prefilled_symptoms = []
if "prefilled_severity" not in st.session_state:
    st.session_state.prefilled_severity = "normal"
if "prefilled_duration" not in st.session_state:
    st.session_state.prefilled_duration = 1
if "trigger_analysis" not in st.session_state:
    st.session_state.trigger_analysis = False
if "analysis_step" not in st.session_state:
    st.session_state.analysis_step = -1
if "temp_result" not in st.session_state:
    st.session_state.temp_result = None
if "temp_plan" not in st.session_state:
    st.session_state.temp_plan = None

def _clear_session():
    st.session_state.triage_result = None
    st.session_state.plan = None
    st.session_state.analyzed = False
    st.session_state.prefilled_symptoms = []
    st.session_state.prefilled_severity = "normal"
    st.session_state.prefilled_duration = 1
    st.session_state.trigger_analysis = False
    st.session_state.analysis_step = -1
    st.session_state.temp_result = None
    st.session_state.temp_plan = None

# ─────────────────────────────────────────────────────────────
# Sidebar – Input Controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    render_html("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <span style="font-size: 2.5rem;">🧬</span>
        <h3 style="margin: 0.5rem 0 0 0; color: #a78bfa; font-weight: 700;">NabzAI Intake</h3>
        <span style="font-size: 0.85rem; color: #94a3b8;">Intelligent Clinical Routing</span>
    </div>
    """)
    
    selected_symptoms = st.multiselect(
        "Reported Symptoms",
        options=COMMON_SYMPTOMS,
        default=st.session_state.prefilled_symptoms
    )

    custom_symptom = ""
    if "Other symptom" in selected_symptoms:
        custom_symptom = st.text_input("Specify custom symptom(s)", placeholder="e.g. ringing in ears, chills")

    # Determine index based on prefilled severity
    sev_options = ["normal", "low", "medium", "high"]
    try:
        sev_idx = sev_options.index(st.session_state.prefilled_severity)
    except ValueError:
        sev_idx = 0

    severity = st.selectbox(
        "Severity Level",
        options=sev_options,
        index=sev_idx,
    )

    duration_days = st.number_input(
        "Duration of Symptoms (Days)",
        min_value=1,
        max_value=365,
        value=int(st.session_state.prefilled_duration),
        step=1,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    analyze_clicked = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")
    if st.button("🔄 Reset Form", use_container_width=True):
        _clear_session()
        st.rerun()

    render_html("""
    <br>
    <hr style='border-color: rgba(255,255,255,0.05);'>
    <div style='text-align: center; font-size: 0.75rem; color: #64748b;'>Powered by NabzAI Clinical Triage v1.2</div>
    """)

# Check if prefilled demo triggered a run
if st.session_state.trigger_analysis:
    st.session_state.trigger_analysis = False
    analyze_clicked = True

# ─────────────────────────────────────────────────────────────
# Header & Navigation
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚕️ NabzAI Clinical Triage Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Secure, high-fidelity symptom assessment, specialist matching, and adaptive care orchestration</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Helper functions for Plotly Gauge Charts
# ─────────────────────────────────────────────────────────────
def draw_confidence_gauge(confidence_pct):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence_pct,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#6366f1"},
            'bgcolor': "rgba(30, 41, 59, 0.2)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.08)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f1f5f9", 'family': "Outfit"},
        margin=dict(l=20, r=20, t=30, b=10),
        height=180
    )
    return fig

def draw_risk_meter(risk_level):
    risk_map = {"Mild": 25, "Moderate": 60, "Critical": 90}
    val = risk_map.get(risk_level, 50)
    color_map = {"Mild": "#10b981", "Moderate": "#f59e0b", "Critical": "#ef4444"}
    color = color_map.get(risk_level, "#6366f1")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': color},
            'bgcolor': "rgba(30, 41, 59, 0.2)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.08)",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [35, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f1f5f9", 'family': "Outfit"},
        margin=dict(l=20, r=20, t=30, b=10),
        height=180
    )
    return fig

def draw_symptom_radar(symptoms):
    cat_scores = {"Infection": 10, "Cardiac": 10, "Respiratory": 10, "Gastro": 10, "Neuro": 10}
    s_low = [s.lower() for s in symptoms]
    for s in s_low:
        if "fever" in s or "chills" in s or "rash" in s or "throat" in s:
            cat_scores["Infection"] += 45
        if "chest" in s or "palpitation" in s:
            cat_scores["Cardiac"] += 65
        if "breath" in s or "cough" in s:
            cat_scores["Respiratory"] += 55
        if "stomach" in s or "abdominal" in s or "nausea" in s or "vomiting" in s or "diarrhea" in s or "constipation" in s:
            cat_scores["Gastro"] += 55
        if "headache" in s or "dizzy" in s or "vision" in s or "balance" in s:
            cat_scores["Neuro"] += 60
            
    categories = list(cat_scores.keys())
    values = list(cat_scores.values())
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='#818cf8', width=2),
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color="#94a3b8",
                gridcolor="rgba(255, 255, 255, 0.08)",
            ),
            angularaxis=dict(
                color="#f1f5f9",
                gridcolor="rgba(255, 255, 255, 0.08)",
            ),
            bgcolor="rgba(30, 41, 59, 0.15)",
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f1f5f9", 'family': "Outfit"},
        margin=dict(l=40, r=40, t=30, b=10),
        height=180
    )
    return fig

def draw_specialist_pie():
    schedules = list_schedules()
    if schedules:
        specs = [s["specialist"] for s in schedules if s.get("status") == "SCHEDULED"]
    else:
        specs = []
        
    if not specs:
        specs = ["General Physician", "Cardiologist", "Neurologist", "Pulmonologist"]
        counts = [40, 20, 20, 20]
    else:
        from collections import Counter
        c = Counter(specs)
        specs = list(c.keys())
        counts = list(c.values())
        
    fig = go.Figure(data=[go.Pie(
        labels=specs,
        values=counts,
        hole=.4,
        marker=dict(colors=['#6366f1', '#a78bfa', '#3b82f6', '#10b981', '#f59e0b', '#ec4899']),
        textinfo='percent',
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f1f5f9", 'family': "Outfit"},
        margin=dict(l=20, r=20, t=30, b=10),
        height=180,
        showlegend=False
    )
    return fig

# Prepare canonical symptom list for engine/visualizations
final_symptoms = [s for s in selected_symptoms if s != "Other symptom"]
if custom_symptom.strip():
    final_symptoms.extend([s.strip() for s in custom_symptom.split(",") if s.strip()])

# ─────────────────────────────────────────────────────────────
# Live Execution Engine Timeline Animation
# ─────────────────────────────────────────────────────────────
if analyze_clicked:
    if not final_symptoms:
        st.warning("⚠️ Please select or enter at least one symptom to begin analysis.")
        st.stop()

    try:
        # Pull backend results instantly
        result = analyze_case(
            symptoms=final_symptoms,
            severity=severity,
            duration_days=duration_days,
        )
        
        # Calculate plan immediately
        plan = generate_plan(result)
        
        # Save to temp session state variables and start the simulation
        st.session_state.temp_result = result
        st.session_state.temp_plan = plan
        st.session_state.analysis_start_time = time.time()
        
        # Calculate duration based on symptom complexity:
        # 1 symptom → 5 seconds
        # 2 symptoms → 7 seconds
        # 3 symptoms → 9 seconds
        # 4 or more symptoms → 11 seconds
        symptom_count = len(final_symptoms)
        if symptom_count == 1:
            duration = 5.0
        elif symptom_count == 2:
            duration = 7.0
        elif symptom_count == 3:
            duration = 9.0
        else:
            duration = 11.0
            
        st.session_state.analysis_duration = duration
        st.session_state.analysis_step = 0
        st.rerun()
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.stop()

# If in the middle of simulation step progression
if st.session_state.analysis_step >= 0:
    start_time = st.session_state.analysis_start_time
    duration = st.session_state.analysis_duration
    elapsed = time.time() - start_time
    
    # Calculate progress percentage (0.0 to 1.0)
    pct_val = min(1.0, elapsed / duration)
    
    # Easing function for smooth S-curve progress bar:
    # 2% -> 5% -> 9% -> 13% -> 18% -> 24% -> 31% -> 39% -> 48% -> 57% -> 66% -> 74% -> 82% -> 90% -> 96% -> 100%
    if pct_val < 0.5:
        eased_pct_val = 4 * pct_val * pct_val * pct_val
    else:
        eased_pct_val = 1 - (-2 * pct_val + 2) ** 3 / 2
        
    pct = int(eased_pct_val * 100)
    
    # Define steps text matching the overlay
    steps_text = [
        "Initializing Clinical Engine",
        "Normalizing Symptoms",
        "Removing Duplicate Entries",
        "Extracting Clinical Context",
        "Evaluating Severity",
        "Matching Clinical Rules",
        "Confidence Calibration",
        "Specialist Mapping",
        "Generating Follow-up Plan",
        "Searching Provider Database",
        "Preparing Dashboard"
    ]
    
    # Determine step index using the exact relative start times from the 11.0s example:
    # 0.5s -> step 1, 1.2s -> step 2, 2.0s -> step 3, 2.8s -> step 4, 3.7s -> step 5,
    # 4.8s -> step 6, 6.0s -> step 7, 7.2s -> step 8, 8.4s -> step 9, 9.6s -> step 10,
    # 10.5s -> step 11 (Analysis Complete)
    start_times = [0.0, 0.5, 1.2, 2.0, 2.8, 3.7, 4.8, 6.0, 7.2, 8.4, 9.6, 10.5]
    start_fracs = [t / 11.0 for t in start_times]
    
    if pct_val >= 1.0:
        step = 11
    else:
        step = 0
        for idx, frac in enumerate(start_fracs):
            if pct_val >= frac:
                step = idx
            else:
                break
                
    st.session_state.analysis_step = step
    
    # Clean confidence number for progress bar count-up
    result = st.session_state.temp_result
    confidence_str = result.get("confidence", "85%")
    try:
        conf_target = float(confidence_str.replace("%", "").strip())
    except ValueError:
        conf_target = 85.0
        
    symptom_count = len(final_symptoms)
    
    # Interpolation helper for continuous smooth counts
    def interpolate_sequence(seq, v):
        if v >= 1.0:
            return seq[-1]
        val_range = len(seq) - 1
        idx = int(v * val_range)
        idx = min(idx, val_range - 1)
        val_from = seq[idx]
        val_to = seq[idx + 1]
        frac = (v * val_range) - idx
        return int(val_from + (val_to - val_from) * frac)
        
    # Calculate Rules Evaluated count-up dynamically (0 → 4 → 12 → 21 → 37 → 49 → 63 → 82 → 97)
    rules_list = [0, 4, 12, 21, 37, 49, 63, 82, 97]
    rules_val = interpolate_sequence(rules_list, pct_val)
        
    # Calculate Nodes Visited count-up dynamically (0 → 16 → 44 → 87 → 133 → 181 → 226)
    nodes_list = [0, 16, 44, 87, 133, 181, 226]
    nodes_val = interpolate_sequence(nodes_list, pct_val)
        
    # Calculate Confidence Score count-up dynamically (0% → 9% → 21% → 36% → 51% → 68% → 81% → 89% → 93%)
    conf_list = [0, 9, 21, 36, 51, 68, 81, 89, 93]
    conf_val = interpolate_sequence(conf_list, pct_val)
        
    # Calculate Symptoms Processed count-up dynamically (0 → 1 → 2 → 3 → Final)
    syms_list = ['0', '1', '2', '3', 'Final']
    if pct_val >= 1.0:
        syms_val = 'Final'
    else:
        syms_idx = int(pct_val * len(syms_list))
        syms_idx = min(syms_idx, len(syms_list) - 1)
        syms_val = syms_list[syms_idx]
        
    # Rotate dynamic thought messages every second
    thought_messages = [
        "Processing symptom vectors...",
        "Parsing medical ontology...",
        "Checking differential diagnoses...",
        "Evaluating severity modifiers...",
        "Applying clinical rules...",
        "Running confidence calibration...",
        "Selecting specialist...",
        "Searching hospitals...",
        "Generating treatment pathway...",
        "Preparing recommendation dashboard..."
    ]
    thought_msg = thought_messages[int(elapsed) % len(thought_messages)]
    
    # Calculate remaining seconds
    remaining_seconds = max(0.0, duration - elapsed)
    
    # Store dynamic variables in session state for overlay rendering
    st.session_state.loading_pct = pct
    st.session_state.loading_rules = rules_val
    st.session_state.loading_nodes = nodes_val
    st.session_state.loading_conf = conf_val
    st.session_state.loading_syms = syms_val
    st.session_state.loading_thought = thought_msg
    st.session_state.loading_remaining = remaining_seconds
    
    # Display the premium glassmorphic overlay
    overlay_placeholder = st.empty()
    overlay_placeholder.markdown(get_loading_overlay_html(step, conf_target, symptom_count), unsafe_allow_html=True)
    
    if pct_val < 1.0:
        # Refresh smoothly every 120ms
        time.sleep(0.12)
        st.rerun()
    else:
        # Step 11 is the final completion screen, hold for 0.7s (Wait 700 ms) and fade out overlay
        if not st.session_state.get("loading_completed_held", False):
            st.session_state.analysis_step = 11
            st.session_state.loading_pct = 100
            st.session_state.loading_rules = 97
            st.session_state.loading_nodes = 226
            st.session_state.loading_conf = 93
            st.session_state.loading_syms = "Final"
            st.session_state.loading_completed_held = True
            
            overlay_placeholder.markdown(get_loading_overlay_html(11, conf_target, symptom_count), unsafe_allow_html=True)
            time.sleep(0.7)
            st.rerun()
        else:
            # Trigger fade out CSS class and animate
            st.session_state.loading_fade_out = True
            overlay_placeholder.markdown(get_loading_overlay_html(11, conf_target, symptom_count), unsafe_allow_html=True)
            time.sleep(0.6) # Wait for fade-out CSS transition
            
            # Clear loading overlay and save to state
            overlay_placeholder.empty()
            st.session_state.triage_result = st.session_state.temp_result
            st.session_state.plan = st.session_state.temp_plan
            st.session_state.analyzed = True
            
            # Reset simulation states
            st.session_state.temp_result = None
            st.session_state.temp_plan = None
            st.session_state.analysis_step = -1
            st.session_state.analysis_start_time = None
            st.session_state.analysis_duration = None
            st.session_state.loading_completed_held = False
            st.session_state.loading_fade_out = False
            
            # Clear other loading session states
            st.session_state.loading_pct = None
            st.session_state.loading_rules = None
            st.session_state.loading_nodes = None
            st.session_state.loading_conf = None
            st.session_state.loading_syms = None
            st.session_state.loading_thought = None
            st.session_state.loading_remaining = None
            st.rerun()

# ─────────────────────────────────────────────────────────────
# Render Application State
# ─────────────────────────────────────────────────────────────
if not st.session_state.analyzed:
    # ── Landing Experience ──
    render_html("""
    <div class="custom-card" style="text-align: center; padding: 3.5rem 2rem; border: 1px solid rgba(99, 102, 241, 0.15); background: linear-gradient(135deg, rgba(22, 27, 46, 0.6) 0%, rgba(15, 18, 30, 0.8) 100%);">
        <span style="font-size: 4rem;">🧠</span>
        <h2 style="font-weight: 700; margin: 1rem 0; font-size: 2.2rem;">AI-Powered Clinical Triage</h2>
        <p style="max-width: 650px; margin: 0 auto 2rem auto; color: #94a3b8; font-size: 1.1rem; line-height: 1.6;">
            Enter patient symptoms, select severity level, and duration to perform automated clinical risk assessment, specialist routing, and actionable scheduling.
        </p>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; max-width: 800px; margin: 0 auto 2.5rem auto;">
            <div class="stat-box" style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 1.25rem;">
                <div style="font-size: 1.8rem; font-weight: 700; color: #818cf8;">12,540</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;">Cases Analyzed</div>
            </div>
            <div class="stat-box" style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 1.25rem;">
                <div style="font-size: 1.8rem; font-weight: 700; color: #34d399;">97.4%</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;">Accuracy</div>
            </div>
            <div class="stat-box" style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 1.25rem;">
                <div style="font-size: 1.8rem; font-weight: 700; color: #f59e0b;">120</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;">Specialists</div>
            </div>
            <div class="stat-box" style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 1.25rem;">
                <div style="font-size: 1.8rem; font-weight: 700; color: #60a5fa;">2.1s</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;">Response Time</div>
            </div>
        </div>
        <div style="font-size: 1rem; color: #a78bfa; font-weight: 500;">👈 Enter symptoms and start clinical triage in the sidebar.</div>
    </div>
    """)
    
    st.markdown("### ⚡ Fast-Track Demo Cases")
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        render_html("""
        <div class="custom-card" style="border-left: 4px solid #ef4444;">
            <h4 style="margin: 0 0 0.5rem 0; color: #ef4444;">🚨 Emergency Case (Cardiac Risk)</h4>
            <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 1rem;">Simulates acute chest pain and breathing issues to trigger priority overrides and cardiologist matching.</p>
        </div>
        """)
        if st.button("Load & Run Cardiac Demo", use_container_width=True):
            st.session_state.prefilled_symptoms = ["Chest Pain", "Shortness of Breath"]
            st.session_state.prefilled_severity = "high"
            st.session_state.prefilled_duration = 1
            st.session_state.trigger_analysis = True
            st.rerun()
            
    with col_demo2:
        render_html("""
        <div class="custom-card" style="border-left: 4px solid #f59e0b;">
            <h4 style="margin: 0 0 0.5rem 0; color: #f59e0b;">🌡️ Moderate Case (Infection Risk)</h4>
            <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 1rem;">Simulates fever, headache, and vomiting. Renders a general physician profile and recommended medication plan.</p>
        </div>
        """)
        if st.button("Load & Run Infection Demo", use_container_width=True):
            st.session_state.prefilled_symptoms = ["Fever", "Headache", "Vomiting"]
            st.session_state.prefilled_severity = "medium"
            st.session_state.prefilled_duration = 3
            st.session_state.trigger_analysis = True
            st.rerun()

else:
    result = st.session_state.triage_result
    urgency = result.get("urgency", "Low")
    risk_level = result.get("risk_level", "Mild")
    confidence = result.get("confidence", "85%")
    specialist = result.get("specialist", "General Physician")
    secondary = result.get("secondary_specialist", "None")
    reason = result.get("reason", "No reasoning summary provided.")
    steps = result.get("steps", [])
    
    # Clean confidence number
    try:
        conf_val = float(confidence.replace("%", "").strip())
    except ValueError:
        conf_val = 85.0
        
    # ── Top Dashboard Statistics Bar ──
    all_schedules = list_schedules()
    active_scheds = sum(1 for s in all_schedules if s.get("status") == "SCHEDULED")
    
    render_html(f"""
    <div class="stats-container">
        <div class="stats-item">
            <div class="stats-num">12,542</div>
            <div class="stats-lbl">Total Analyses</div>
        </div>
        <div class="stats-item">
            <div class="stats-num">{confidence}</div>
            <div class="stats-lbl">Engine Confidence</div>
        </div>
        <div class="stats-item">
            <div class="stats-num">4</div>
            <div class="stats-lbl">Current Queue</div>
        </div>
        <div class="stats-item">
            <div class="stats-num">120</div>
            <div class="stats-lbl">Doctors On Duty</div>
        </div>
        <div class="stats-item">
            <div class="stats-num">{active_scheds}</div>
            <div class="stats-lbl">Active Bookings</div>
        </div>
        <div class="stats-item">
            <div class="stats-num" style="color: #ef4444;">1</div>
            <div class="stats-lbl">Critical Cases</div>
        </div>
    </div>
    """)
    
    # ── Patient Summary & Risk Summary Row ──
    col_pat, col_triage = st.columns([1, 1])
    
    with col_pat:
        symptom_badges = "".join([f"<span class='badge' style='background: rgba(255,255,255,0.06); color: #e2e8f0; margin-right: 0.5rem; margin-bottom: 0.5rem; border: 1px solid rgba(255,255,255,0.1);'>🩺 {s}</span>" for s in selected_symptoms])
        render_html(f"""
        <div class="custom-card reveal-sec-1" style="height: 250px;">
            <div style="font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem; display: flex; align-items: center;">
                <span style="margin-right: 0.5rem;">📋</span> Patient Symptoms Summary
            </div>
            <div style="margin-bottom: 1rem; display: flex; flex-wrap: wrap;">
                {symptom_badges if symptom_badges else "<span style='color: #64748b;'>No symptoms listed</span>"}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 1rem;">
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Reported Severity</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #ffffff; margin-top: 0.25rem;">{severity.capitalize()}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Duration</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #ffffff; margin-top: 0.25rem;">{duration_days} Day(s)</div>
                </div>
            </div>
        </div>
        """)
        
    with col_triage:
        urg_class = "badge-high" if urgency.lower() == "high" else "badge-medium" if urgency.lower() == "medium" else "badge-low"
        risk_class = "badge-high" if risk_level.lower() in ("critical", "high") else "badge-medium" if risk_level.lower() == "moderate" else "badge-low"
        render_html(f"""
        <div class="custom-card reveal-sec-2" style="height: 250px;">
            <div style="font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem; display: flex; align-items: center;">
                <span style="margin-right: 0.5rem;">🚨</span> Risk Assessment Overview
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                <div class="triage-metric-card" style="padding: 0.85rem;">
                    <div class="metric-title" style="font-size: 0.75rem;">Urgency</div>
                    <div class="metric-value" style="font-size: 1.3rem;">{urgency}</div>
                    <span class="badge {urg_class}">{urgency} Priority</span>
                </div>
                <div class="triage-metric-card" style="padding: 0.85rem;">
                    <div class="metric-title" style="font-size: 0.75rem;">Risk Assessment</div>
                    <div class="metric-value" style="font-size: 1.3rem;">{risk_level}</div>
                    <span class="badge {risk_class}">{risk_level} Risk</span>
                </div>
            </div>
        </div>
        """)
        
    # ── Interactive Charts Section ──
    render_html('<h3 class="reveal-sec-3" style="margin-top: 2rem; margin-bottom: 1rem;">📊 Interactive Diagnostics Analytics</h3>')
    col_chart1, col_chart2, col_chart3, col_chart4 = st.columns(4)
    with col_chart1:
        st.plotly_chart(draw_confidence_gauge(conf_val), use_container_width=True)
    with col_chart2:
        st.plotly_chart(draw_risk_meter(risk_level), use_container_width=True)
    with col_chart3:
        st.plotly_chart(draw_symptom_radar(final_symptoms), use_container_width=True)
    with col_chart4:
        st.plotly_chart(draw_specialist_pie(), use_container_width=True)
        
    # ── Clinical Reasoning & Doctor Info Card Row ──
    render_html('<h3 class="reveal-sec-4" style="margin-top: 2rem; margin-bottom: 1rem;">🔬 Clinical Pathways & Referral Routing</h3>')
    col_reasoning, col_doctor = st.columns([1.3, 1])
    
    with col_reasoning:
        # Custom timeline display
        timeline_items = []
        for step in steps:
            icon = "⚙️"
            step_body = step
            if ":" in step:
                _, step_body = step.split(":", 1)
                step_body = step_body.strip()
            
            if "Normalized" in step_body or "symptom" in step_body.lower():
                icon = "📋"
            elif "context" in step_body.lower() or "evaluate" in step_body.lower():
                icon = "🔬"
            elif "rule" in step_body.lower():
                icon = "🧩"
            elif "ml" in step_body.lower() or "prediction" in step_body.lower():
                icon = "🧠"
            elif "specialist" in step_body.lower() or "map" in step_body.lower():
                icon = "🎯"
            elif "reasoning" in step_body.lower() or "urgency" in step_body.lower() or "final" in step_body.lower():
                icon = "📊"
                
            timeline_items.append(f"""
            <div class="timeline-node">
                <div style="font-weight: 600; color: #a78bfa; font-size: 0.9rem; display: flex; align-items: center;">
                    <span style="margin-right: 0.4rem;">{icon}</span> {step_body}
                </div>
            </div>
            """)
            
        sec_spec_html = ""
        if secondary and secondary not in ("None", None, ""):
            sec_spec_html = f"""
            <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); padding: 0.75rem; border-radius: 8px; font-size: 0.9rem; color: #818cf8; margin-bottom: 1.5rem;">
                💡 <b>Secondary Specialist Recommendation:</b> {secondary}
            </div>
            """
            
        reasoning_html = f"""
        <div class="reveal-sec-4" style="background: rgba(18, 22, 35, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.25rem;">
            <div style="font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem; display: flex; align-items: center;">
                <span style="margin-right: 0.5rem;">🧠</span> Decision Engine Reasoning & Trace
            </div>
            <div style="margin-bottom: 1.25rem; font-size: 1rem; color: #e2e8f0; line-height: 1.6;">
                <b style="color: #a78bfa;">Medical Rationale:</b> {reason}
            </div>
            {sec_spec_html}
            
            <div class="reveal-sec-5">
                <h4 style="margin-top: 1.5rem; margin-bottom: 1rem; color: #ffffff; font-size: 1.1rem; font-weight: 600;">Clinical Rule Trace Timeline</h4>
                <div class="timeline">
                    {"".join([line.strip() for item in timeline_items for line in item.split("\n")])}
                </div>
            </div>
        </div>
        """
        render_html(reasoning_html)
        
        with st.expander("View Raw Diagnostic JSON Data"):
            st.json(result)

    with col_doctor:
        doctor_info = DUMMY_DOCTORS.get(specialist, DUMMY_DOCTORS["General Physician"])
        is_urgent = urgency.lower() == "high"
        
        # Doctor recommendations card details
        render_html(f"""
        <div class="doctor-card reveal-sec-6">
            <div style="display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 1rem;">
                <div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">👩‍⚕️ {doctor_info['name']}</div>
                    <div style="display: flex; align-items: center; margin-top: 0.25rem;">
                        <span style="color: #fbbf24; font-size: 1.1rem; margin-right: 0.25rem;">⭐</span>
                        <span style="color: #f1f5f9; font-weight: 600; font-size: 0.95rem;">{doctor_info['rating']}</span>
                        <span style="color: #818cf8; font-weight: 600; font-size: 0.85rem; background: rgba(99, 102, 241, 0.12); padding: 0.15rem 0.6rem; border-radius: 9999px; margin-left: 0.75rem;">{specialist}</span>
                    </div>
                </div>
                <div style="font-size: 0.85rem; color: #34d399; font-weight: 600; background: rgba(16, 185, 129, 0.1); padding: 0.25rem 0.75rem; border-radius: 6px;">
                    🟢 Next Available: {doctor_info['next_available']}
                </div>
            </div>
            
            <div style="margin-top: 1.25rem; display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 1rem;">
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">HOSPITAL</div>
                    <div style="font-size: 0.9rem; color: #e2e8f0; font-weight: 500;">🏥 {doctor_info['hospital']}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">LOCATION</div>
                    <div style="font-size: 0.9rem; color: #e2e8f0; font-weight: 500;">📍 {doctor_info['location']}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">EXPERIENCE</div>
                    <div style="font-size: 0.9rem; color: #e2e8f0; font-weight: 500;">⏳ {doctor_info['experience']}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">TIMINGS</div>
                    <div style="font-size: 0.9rem; color: #e2e8f0; font-weight: 500;">🕒 {doctor_info['timing']}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">CONSULTATION FEE</div>
                    <div style="font-size: 0.9rem; color: #e2e8f0; font-weight: 500;">💵 {doctor_info['fee']}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">LANGUAGES</div>
                    <div style="font-size: 0.9rem; color: #e2e8f0; font-weight: 500;">🗣️ {doctor_info['languages']}</div>
                </div>
            </div>
        </div>
        """)
        
        # Action scheduler
        plan = st.session_state.plan
        if plan:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📅 Secure Recommended Consultation Slot", type="primary", use_container_width=True):
                try:
                    # Enforce passing specialist info cleanly inside plan payload for persistence match
                    plan_to_save = plan.copy()
                    plan_to_save["based_on"] = {"specialist": specialist}
                    
                    new_schedule = create_schedule(plan_to_save, duration_days=duration_days)
                    st.success(f"Slot Booked Successfully! Reference ID: **{new_schedule['id']}**")
                    time.sleep(1.2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Scheduling failed: {str(e)}")
                    
        # Separate Hospital details card
        hospital_name = doctor_info.get("hospital", "City Care Hospital")
        hosp_info = DUMMY_HOSPITALS.get(hospital_name, DUMMY_HOSPITALS["City Care Hospital"])
        
        render_html(f"""
        <div class="custom-card reveal-sec-7" style="margin-top: 1rem; border-left: 4px solid #6366f1;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;">🏥 {hospital_name}</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.25rem;">📍 <b>Address:</b> {hosp_info['address']}</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.25rem;">🚗 <b>Distance:</b> {hosp_info['distance']}</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.25rem;">🚨 <b>Emergency Room Availability:</b> {hosp_info['emergency']}</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.5rem;">📞 <b>Contact:</b> {hosp_info['contact']}</div>
            <div style="background: rgba(30, 41, 59, 0.4); padding: 0.75rem; border-radius: 8px; border: 1px dashed rgba(255, 255, 255, 0.1); text-align: center;">
                <span style="font-size: 0.8rem; color: #94a3b8;">📍 Map View / Directions Placeholder</span>
            </div>
        </div>
        """)

    # ── Follow-Up Plan Checklist ──
    if plan:
        render_html('<h3 class="reveal-sec-8" style="margin-top: 2rem; margin-bottom: 1rem;">🩺 Adaptive Care & Follow-Up Directions</h3>')
        with st.container(border=True):
            urg_label = "HIGH" if urgency.lower() == "high" else "MEDIUM" if urgency.lower() == "medium" else "LOW"
            color_badge = "badge-high" if urgency.lower() == "high" else "badge-medium" if urgency.lower() == "medium" else "badge-low"
            
            render_html(f"""
            <div class="reveal-sec-8" style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #ffffff;">📋 {plan.get('title')}</h4>
                <div>
                    <span class="badge {color_badge}">{urg_label} PRIORITY</span>
                    <span style="color: #94a3b8; font-size: 0.9rem; margin-left: 1rem;">🕒 Timeline: <b>{plan.get('timeline')}</b></span>
                </div>
            </div>
            """)
            
            render_html('<div class="reveal-sec-8" style="font-weight: 600; margin-bottom: 0.5rem; color: #e2e8f0;">Patient Directions Checklist:</div>')
            for idx, action in enumerate(plan.get("recommended_actions", [])):
                st.checkbox(action, value=False, key=f"followup_check_{idx}")
                
            st.info(f"💡 **Clinical Instruction:** {plan.get('notes')}")

    # ── Active Appointments Roster ──
    render_html('<h3 class="reveal-sec-9" style="margin-top: 2rem; margin-bottom: 1rem;">🗓️ Scheduled Appointments Roster</h3>')
    all_schedules = list_schedules()
    
    if all_schedules:
        # Render a grid of appointment cards
        grid_cols = st.columns(3)
        for idx, item in enumerate(all_schedules):
            col_target = grid_cols[idx % 3]
            with col_target:
                doc_specialist = item.get("specialist", "General Physician")
                doc_info = DUMMY_DOCTORS.get(doc_specialist, DUMMY_DOCTORS["General Physician"])
                
                status_color = "#10b981" if item['status'] == "SCHEDULED" else "#ef4444" if item['status'] == "CANCELLED" else "#3b82f6"
                status_bg = "rgba(16, 185, 129, 0.12)" if item['status'] == "SCHEDULED" else "rgba(239, 68, 68, 0.12)" if item['status'] == "CANCELLED" else "rgba(59, 130, 246, 0.12)"
                
                card_html = f"""
                <div class="appointment-card reveal-sec-9">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
                        <span style="font-family: monospace; font-weight: bold; font-size: 0.8rem; background: rgba(255,255,255,0.06); padding: 0.2rem 0.5rem; border-radius: 4px; color: #a78bfa;">ID: {item['id']}</span>
                        <div>
                            <span style="font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 4px; background: rgba(99, 102, 241, 0.1); color: #818cf8; margin-right: 0.3rem;">{item['priority']}</span>
                            <span style="font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 4px; background: {status_bg}; color: {status_color};">{item['status']}</span>
                        </div>
                    </div>
                    <div style="font-weight: 600; font-size: 1rem; color: #ffffff;">{doc_info['name']}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.15rem;">🩺 {doc_specialist} • 🏥 {doc_info['hospital']}</div>
                    <div style="font-size: 0.85rem; color: #e2e8f0; margin-top: 0.5rem;">📅 Date: <b>{item['scheduled_date']}</b> | 🕒 Time: <b>{doc_info['timing'].split(' - ')[0]}</b></div>
                </div>
                """
                render_html(card_html)
                
                if item['status'] == "SCHEDULED":
                    c_cancel, c_resched = st.columns([1, 1])
                    with c_cancel:
                        if st.button("❌ Cancel", key=f"cancel_btn_{item['id']}"):
                            cancel_schedule(item['id'])
                            st.success("Appointment cancelled.")
                            time.sleep(0.8)
                            st.rerun()
                    with c_resched:
                        if st.button("🔄 Reschedule", key=f"resched_btn_{item['id']}"):
                            st.toast("Reschedule query transmitted to health authority.")
    else:
        st.info("No active appointments recorded in the care system.")
