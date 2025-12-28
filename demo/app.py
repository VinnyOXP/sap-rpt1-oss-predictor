"""
SAP-RPT-1-OSS Predictor - Claude Skill Demo
This app demonstrates how Claude uses the SAP-RPT-1-OSS skill for predictive analytics.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from anthropic import Anthropic

# Page config
st.set_page_config(
    page_title="Claude + SAP-RPT-1-OSS Skill Demo",
    page_icon="🤖",
    layout="wide"
)

# Load SKILL.md content
@st.cache_data
def load_skill():
    skill_path = Path(__file__).parent.parent / "SKILL.md"
    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8")
    return None

SKILL_CONTENT = load_skill()

# Header
st.title("🤖 Claude + SAP-RPT-1-OSS Skill Demo")
st.markdown("""
This demo showcases how **Claude uses the SAP-RPT-1-OSS skill** to help with predictive analytics on SAP data.

The skill teaches Claude how to:
- Set up and use the SAP-RPT-1-OSS tabular foundation model
- Prepare SAP data for predictions
- Run classification and regression tasks
- Handle batch processing for large datasets
""")

# Sidebar - API Configuration
st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "Claude API Key",
    type="password",
    placeholder="sk-ant-...",
    help="Get your API key from console.anthropic.com"
)

if not api_key:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

model = st.sidebar.selectbox(
    "Model",
    ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
    index=0
)

st.sidebar.markdown("---")

# Show skill content
with st.sidebar.expander("📜 View SKILL.md Content"):
    if SKILL_CONTENT:
        st.code(SKILL_CONTENT[:2000] + "\n\n... (truncated)", language="markdown")
    else:
        st.warning("SKILL.md not found")

st.sidebar.markdown("""
### Resources
- [Skill Repository](https://github.com/amitlals/sap-rpt1-oss-predictor)
- [SAP-RPT-1-OSS](https://github.com/SAP-samples/sap-rpt-1-oss)
- [Claude API Docs](https://docs.anthropic.com)
""")

# Main content
col1, col2 = st.columns([2, 1])

with col2:
    st.header("📊 Upload Data (Optional)")
    uploaded_file = st.file_uploader("Upload CSV for context", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows")
        st.dataframe(df.head(5), use_container_width=True)
        data_context = f"\n\nUser's uploaded data preview:\n```\n{df.head(10).to_string()}\n```\nColumns: {list(df.columns)}"
    else:
        data_context = ""
        df = None
    
    # Sample prompts
    st.header("💡 Try These Prompts")
    
    sample_prompts = [
        "How do I set up SAP-RPT-1-OSS for customer churn prediction?",
        "What GPU do I need to run RPT-1 locally?",
        "Show me how to predict payment default risk from SAP FI-AR data",
        "How do I prepare my SAP data for the RPT-1 model?",
        "What's the difference between classification and regression in RPT-1?",
        "Generate Python code to predict delivery delays using my data"
    ]
    
    for prompt in sample_prompts:
        if st.button(prompt, key=prompt, use_container_width=True):
            st.session_state.selected_prompt = prompt

with col1:
    st.header("💬 Chat with Claude (using SAP-RPT-1-OSS Skill)")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get prompt from button or input
    if "selected_prompt" in st.session_state:
        prompt = st.session_state.selected_prompt
        del st.session_state.selected_prompt
    else:
        prompt = st.chat_input("Ask Claude about SAP-RPT-1-OSS predictions...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            if not api_key:
                st.error("""
                ⚠️ **API Key Required**
                
                Please enter your Claude API key in the sidebar, or set the `ANTHROPIC_API_KEY` environment variable.
                
                Get your key at: [console.anthropic.com](https://console.anthropic.com)
                """)
                response = "API key required to continue."
            else:
                with st.spinner("Claude is thinking..."):
                    try:
                        client = Anthropic(api_key=api_key)
                        
                        # Build system prompt with skill
                        system_prompt = f"""You are an expert assistant with deep knowledge of SAP systems and predictive analytics.

You have been given the following skill that teaches you how to use the SAP-RPT-1-OSS tabular foundation model:

<skill>
{SKILL_CONTENT}
</skill>

Use this skill to help users with:
- Setting up SAP-RPT-1-OSS model
- Preparing SAP data for predictions
- Running classification (churn, default risk) and regression (delay days, demand)
- Understanding hardware requirements
- Writing Python code for predictions

Always reference the skill's guidance and provide practical, actionable advice.
When showing code, use the patterns from the skill.
{data_context}
"""
                        
                        # Build messages
                        api_messages = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                        
                        # Call Claude API
                        response_obj = client.messages.create(
                            model=model,
                            max_tokens=4096,
                            system=system_prompt,
                            messages=api_messages
                        )
                        
                        response = response_obj.content[0].text
                        st.markdown(response)
                        
                    except Exception as e:
                        response = f"❌ Error: {str(e)}"
                        st.error(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")

# How it works section
with st.expander("🔍 How This Demo Works"):
    st.markdown("""
    ### Claude Skill Integration
    
    This demo shows the **exact pattern** for using Claude skills:
    
    1. **SKILL.md is loaded** as context in the system prompt
    2. **Claude reads the skill** and gains expertise about SAP-RPT-1-OSS
    3. **User asks questions** about predictions, setup, or code
    4. **Claude responds** using the skill's guidance
    
    ### Code Flow
    
    ```python
    # 1. Load the skill
    skill_content = Path("SKILL.md").read_text()
    
    # 2. Create system prompt with skill
    system_prompt = f\"\"\"
    You have this skill:
    <skill>
    {skill_content}
    </skill>
    Use it to help users...
    \"\"\"
    
    # 3. Call Claude API
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=system_prompt,
        messages=[{"role": "user", "content": user_question}]
    )
    ```
    
    ### Why Skills Matter
    
    - **Instant expertise**: Claude immediately knows SAP-RPT-1-OSS
    - **Consistent guidance**: Always follows skill's best practices
    - **Code generation**: Produces correct Python patterns
    - **Reusable**: Same skill works across Claude.ai, API, Claude Code
    """)

st.markdown("""
<div style='text-align: center; color: gray;'>
    <a href="https://github.com/amitlals/sap-rpt1-oss-predictor">SAP-RPT-1-OSS Predictor Skill</a> | 
    Powered by Claude | Apache 2.0 License
</div>
""", unsafe_allow_html=True)
