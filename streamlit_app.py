#!/usr/bin/env python3
"""
RepoSpeak - Streamlit Dashboard
Interactive web interface for repository analysis with chat and visualizations

Usage: streamlit run streamlit_app.py
"""

import streamlit as st
import json
from pathlib import Path
import anthropic
import os
from datetime import datetime
import config

# Page configuration
st.set_page_config(
    page_title="RepoSpeak - Code Analysis Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .dead-code-high {
        background-color: #ffebee;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .dead-code-medium {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


def load_context_files():
    """Load all available context files"""
    contexts = {}
    for ctx_file in Path(".").glob("context_*.json"):
        repo_name = ctx_file.stem.replace("context_", "")
        contexts[repo_name] = str(ctx_file)
    return contexts


def load_context(context_file):
    """Load context data from JSON file"""
    with open(context_file, 'r') as f:
        return json.load(f)


def display_overview(context):
    """Display repository overview"""
    st.markdown('<p class="main-header"> Repository Overview</p>', unsafe_allow_html=True)

    repo_name = context.get("repo_name", "Unknown")
    repo_summary = context.get("repo_summary", {})

    # Stats row
    col1, col2, col3 = st.columns(3)

    ast_results = context.get("ast_results", {})
    file_count = len([k for k in ast_results.keys() if k != '__analysis_summary__'])

    total_functions = 0
    total_classes = 0
    for module, data in ast_results.items():
        if module != '__analysis_summary__':
            total_functions += len(data.get('functions', []))
            total_classes += len(data.get('classes', []))

    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric(" Files", file_count)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric(" Functions", total_functions)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric(" Classes", total_classes)
        st.markdown('</div>', unsafe_allow_html=True)

    # Repository Summary
    st.markdown("---")
    st.subheader(" Repository Summary")

    if repo_summary and 'human' in repo_summary:
        st.write(repo_summary['human'])
    else:
        st.info("No summary available")

    # Technical Summary (collapsible)
    if repo_summary and 'technical' in repo_summary:
        with st.expander(" Technical Summary"):
            st.write(repo_summary['technical'])


def display_audio(context):
    """Display audio narration"""
    st.markdown('<p class="main-header"> Audio Narration</p>', unsafe_allow_html=True)

    repo_name = context.get("repo_name", "Unknown")
    audio_file = Path(f"audio/{repo_name}_summary.mp3")
    script_file = Path(f"audio/{repo_name}_audio_script.txt")

    if audio_file.exists():
        # Audio info
        audio_info = context.get("audio_narration", {})
        if audio_info:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(" Word Count", audio_info.get('word_count', 'N/A'))
            with col2:
                st.metric(" Duration", f"~{audio_info.get('estimated_duration_minutes', 'N/A')} min")

        st.markdown("---")

        # Audio player
        st.subheader(" Play Audio Summary")
        with open(audio_file, 'rb') as audio:
            audio_bytes = audio.read()
            st.audio(audio_bytes, format='audio/mp3')

        # Script preview
        if script_file.exists():
            st.markdown("---")
            st.subheader(" Audio Script")
            with open(script_file, 'r') as f:
                script_content = f.read()
                st.text_area("Script", script_content, height=300)
    else:
        st.warning("⚠️ No audio narration found")
        st.info(f"Generate audio by running: `python3 generate_audio.py {repo_name}`")


def display_architecture(context):
    """Display architecture diagram"""
    st.markdown('<p class="main-header"> Architecture Diagram</p>', unsafe_allow_html=True)

    repo_name = context.get("repo_name", "Unknown")
    mermaid_file = Path(f"diagrams/mermaid_{repo_name}.mmd")

    if mermaid_file.exists():
        with open(mermaid_file, 'r') as f:
            mermaid_code = f.read()

        # Create mermaid.live link with pre-loaded diagram
        import base64
        import json

        state = {
            "code": mermaid_code,
            "mermaid": {"theme": "default"},
            "autoSync": True,
            "updateDiagram": True
        }
        state_json = json.dumps(state)
        encoded_state = base64.b64encode(state_json.encode('utf-8')).decode('utf-8')
        mermaid_link = f"https://mermaid.live/edit#base64:{encoded_state}"

        # Display Mermaid diagram
        st.subheader(" Visual Architecture")

        # Add clickable link button
        st.markdown(f" **[Click here to view interactive diagram in Mermaid Live]({mermaid_link})**")
        st.markdown("---")

        st.code(mermaid_code, language="mermaid")

        # Logical groups
        logical_groups = context.get("logical_groups", {})
        if logical_groups:
            st.markdown("---")
            st.subheader(" Logical Groups")
            for group_name, files in logical_groups.items():
                with st.expander(f"{group_name} ({len(files)} files)"):
                    for file in sorted(files):
                        st.write(f"• {file}")
    else:
        st.warning("⚠️ No architecture diagram found")


def display_dependencies(context):
    """Display module dependencies"""
    st.markdown('<p class="main-header"> Module Dependencies</p>', unsafe_allow_html=True)

    repo_name = context.get("repo_name", "Unknown")
    deps_file = Path(f"diagrams/dependencies_{repo_name}.txt")

    if deps_file.exists():
        with open(deps_file, 'r') as f:
            deps_content = f.read()

        st.code(deps_content, language="text")
    else:
        st.warning("⚠️ No dependency graph found")


def display_dead_code(context):
    """Display dead code analysis"""
    st.markdown('<p class="main-header"> Dead Code Analysis</p>', unsafe_allow_html=True)

    dead_code_analysis = context.get("dead_code_analysis", {})

    if not dead_code_analysis:
        st.warning("⚠️ No dead code analysis found")
        return

    # Get all categories of dead code (excluding unreachable_code)
    unreferenced_functions = dead_code_analysis.get("unreferenced_functions", [])
    unused_classes = dead_code_analysis.get("unused_classes", [])
    unused_imports = dead_code_analysis.get("unused_imports", [])
    unused_globals = dead_code_analysis.get("unused_global_variables", [])
    suspicious_patterns = dead_code_analysis.get("suspicious_patterns", [])

    # Summary stats
    total_dead = (len(unreferenced_functions) + len(unused_classes) +
                  len(unused_imports) + len(unused_globals) +
                  len(suspicious_patterns))

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(" Total", total_dead)
    with col2:
        st.metric(" Functions", len(unreferenced_functions))
    with col3:
        st.metric(" Classes", len(unused_classes))
    with col4:
        st.metric(" Imports", len(unused_imports))
    with col5:
        st.metric(" Globals", len(unused_globals))
    with col6:
        st.metric(" Patterns", len(suspicious_patterns))

    st.markdown("---")

    # Display summary if available
    # summary = dead_code_analysis.get("summary", {})
    # if summary:
    #     st.subheader(" Summary")
    #     st.write(summary.get("overview", "No summary available"))

    st.markdown("---")

    # Display each category
    if unreferenced_functions:
        with st.expander(f" Unreferenced Functions ({len(unreferenced_functions)})", expanded=True):
            for func in unreferenced_functions:
                st.markdown(f"**`{func.get('name', 'Unknown')}`**")

                # Module name
                module_name = func.get('module') or func.get('file', 'Unknown')
                st.caption(f" **Module:** {module_name} : Line {func.get('lineno', '?')}")

                # LLM Analysis
                llm_analysis = func.get('llm_analysis', {})
                status = func.get('status', 'unknown')
                recommendation = func.get('recommendation', 'unknown')
                confidence = func.get('confidence', 0)
                reason = llm_analysis.get('reason', func.get('reason', 'Not referenced by any code'))

                # Status with color coding
                status_colors = {
                    'dead_code': '🔴',
                    'false_positive': '🟢',
                    'uncertain': '🟡'
                }
                status_icon = status_colors.get(status, '⚪')

                st.write(f"**Status:** {status_icon} {status}")
                st.write(f"**Confidence:** {confidence}%")
                st.write(f"**Recommendation:** {recommendation}")
                st.write(f"**Reason:** {reason}")
                st.markdown("---")

    if unused_classes:
        with st.expander(f" Unreferenced Classes ({len(unused_classes)})", expanded=False):
            for cls in unused_classes:
                st.markdown(f"**`{cls.get('name', 'Unknown')}`**")

                # Module name
                module_name = cls.get('module') or cls.get('file', 'Unknown')
                st.caption(f" **Module:** {module_name} : Line {cls.get('lineno', '?')}")

                # LLM Analysis
                llm_analysis = cls.get('llm_analysis', {})
                status = cls.get('status', 'unknown')
                recommendation = cls.get('recommendation', 'unknown')
                confidence = cls.get('confidence', 0)
                reason = llm_analysis.get('reason', cls.get('reason', 'Not referenced by any code'))

                # Status with color coding
                status_colors = {
                    'dead_code': '🔴',
                    'false_positive': '🟢',
                    'uncertain': '🟡'
                }
                status_icon = status_colors.get(status, '⚪')

                st.write(f"**Status:** {status_icon} {status}")
                st.write(f"**Confidence:** {confidence}%")
                st.write(f"**Recommendation:** {recommendation}")
                st.write(f"**Reason:** {reason}")
                st.markdown("---")

    if unused_imports:
        with st.expander(f" Unreferenced Imports ({len(unused_imports)})", expanded=False):
            for imp in unused_imports:
                st.markdown(f"**`{imp.get('name', 'Unknown')}`**")

                # Module name
                module_name = imp.get('module') or imp.get('file', 'Unknown')
                st.caption(f" **Module:** {module_name} : Line {imp.get('lineno', '?')}")

                # LLM Analysis
                llm_analysis = imp.get('llm_analysis', {})
                status = imp.get('status', 'unknown')
                recommendation = imp.get('recommendation', 'unknown')
                confidence = imp.get('confidence', 0)
                reason = llm_analysis.get('reason', imp.get('reason', 'Imported but never used'))

                # Status with color coding
                status_colors = {
                    'dead_code': '🔴',
                    'false_positive': '🟢',
                    'uncertain': '🟡'
                }
                status_icon = status_colors.get(status, '⚪')

                st.write(f"**Status:** {status_icon} {status}")
                st.write(f"**Confidence:** {confidence}%")
                st.write(f"**Recommendation:** {recommendation}")
                st.write(f"**Reason:** {reason}")
                st.markdown("---")

    if unused_globals:
        with st.expander(f" Unreferenced Global Variables ({len(unused_globals)})", expanded=False):
            for glob in unused_globals:
                st.markdown(f"**`{glob.get('name', 'Unknown')}`**")

                # Module name
                module_name = glob.get('module') or glob.get('file', 'Unknown')
                st.caption(f" **Module:** {module_name} : Line {glob.get('lineno', '?')}")

                # LLM Analysis
                llm_analysis = glob.get('llm_analysis', {})
                status = glob.get('status', 'unknown')
                recommendation = glob.get('recommendation', 'unknown')
                confidence = glob.get('confidence', 0)
                reason = llm_analysis.get('reason', glob.get('reason', 'Global variable never used'))

                # Status with color coding
                status_colors = {
                    'dead_code': '🔴',
                    'false_positive': '🟢',
                    'uncertain': '🟡'
                }
                status_icon = status_colors.get(status, '⚪')

                st.write(f"**Status:** {status_icon} {status}")
                st.write(f"**Confidence:** {confidence}%")
                st.write(f"**Recommendation:** {recommendation}")
                st.write(f"**Reason:** {reason}")
                st.markdown("---")

    if suspicious_patterns:
        with st.expander(f" Suspicious Patterns ({len(suspicious_patterns)})", expanded=False):
            for pattern in suspicious_patterns:
                st.markdown(f"**`{pattern.get('name', 'Unknown')}`**")

                # Module name
                module_name = pattern.get('module') or pattern.get('file', 'Unknown')
                st.caption(f" **Module:** {module_name} : Line {pattern.get('lineno', '?')}")

                # LLM Analysis
                llm_analysis = pattern.get('llm_analysis', {})
                status = pattern.get('status', 'unknown')
                recommendation = pattern.get('recommendation', 'unknown')
                confidence = pattern.get('confidence', 0)
                reason = llm_analysis.get('reason', pattern.get('reason', 'No description'))

                # Status with color coding
                status_colors = {
                    'dead_code': '🔴',
                    'false_positive': '🟢',
                    'uncertain': '🟡'
                }
                status_icon = status_colors.get(status, '⚪')

                st.write(f"**Pattern Type:** {pattern.get('pattern_type', 'Unknown')}")
                st.write(f"**Status:** {status_icon} {status}")
                st.write(f"**Confidence:** {confidence}%")
                st.write(f"**Recommendation:** {recommendation}")
                st.write(f"**Reason:** {reason}")
                st.markdown("---")

    if total_dead == 0:
        st.success(" No dead code detected! Your codebase looks clean.")


def display_modules(context):
    """Display all modules with summaries"""
    st.markdown('<p class="main-header"> Modules</p>', unsafe_allow_html=True)

    module_summaries = context.get("module_summaries", {})

    if not module_summaries:
        st.warning("⚠️ No module summaries found")
        return

    # Search box
    search = st.text_input(" Search modules", placeholder="e.g., auth, storage, validate")

    for module_name, summary in sorted(module_summaries.items()):
        # Filter by search
        if search and search.lower() not in module_name.lower():
            continue

        with st.expander(f" {module_name}"):
            if 'human' in summary:
                st.write(summary['human'])

            if 'technical' in summary:
                with st.expander(" Technical Details"):
                    st.write(summary['technical'])


def display_functions(context):
    """Display all functions with summaries"""
    st.markdown('<p class="main-header"> Functions</p>', unsafe_allow_html=True)

    function_summaries = context.get("function_summaries", {})

    if not function_summaries:
        st.warning("⚠️ No function summaries found")
        return

    # Search box
    search = st.text_input(" Search functions", placeholder="e.g., validate, save, process")

    # Count total functions
    total_functions = sum(len(funcs) for funcs in function_summaries.values())
    st.caption(f"Total functions: {total_functions}")

    # Display by module
    for module_name, functions in sorted(function_summaries.items()):
        # Check if any function matches search
        if search:
            matching_funcs = {name: summary for name, summary in functions.items()
                            if search.lower() in name.lower()}
            if not matching_funcs:
                continue
            functions = matching_funcs

        with st.expander(f" {module_name} ({len(functions)} functions)"):
            for func_name, summary in sorted(functions.items()):
                st.markdown(f"### `{func_name}()`")

                if isinstance(summary, dict):
                    if 'human' in summary:
                        st.write(summary['human'])

                    if 'technical' in summary:
                        with st.expander(" Technical Details"):
                            st.write(summary['technical'])
                else:
                    st.write(summary)

                st.markdown("---")


def chat_interface(context):
    """Chat interface for asking questions"""
    st.markdown('<p class="main-header"> Ask Questions</p>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Check if message contains audio data
            if isinstance(message["content"], dict) and message["content"].get("type") == "audio":
                audio_data = message["content"]
                st.markdown("** Audio Narration:**\n")

                # Show audio info
                audio_info = audio_data.get("audio_info", {})
                if audio_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(" Word Count", audio_info.get('word_count', 'N/A'))
                    with col2:
                        st.metric(" Duration", f"~{audio_info.get('estimated_duration_minutes', 'N/A')} min")

                # Display audio player
                audio_file = Path(audio_data["audio_file"])
                if audio_file.exists():
                    with open(audio_file, 'rb') as f:
                        audio_bytes = f.read()
                        st.audio(audio_bytes, format='audio/mp3')

                # Show script
                script_file = audio_data.get("script_file")
                if script_file and Path(script_file).exists():
                    st.markdown("** Audio Script:**\n")
                    with open(script_file, 'r') as f:
                        script_content = f.read()
                        with st.expander("Click to view full transcript"):
                            st.text_area("Transcript", script_content, height=300, key=f"script_history_{idx}", disabled=True)
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about this codebase"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_chat_response(prompt, context)

                # Check if response is audio data
                if isinstance(response, dict) and response.get("type") == "audio":
                    audio_data = response
                    st.markdown("** Audio Narration:**\n")

                    # Show audio info
                    audio_info = audio_data.get("audio_info", {})
                    if audio_info:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(" Word Count", audio_info.get('word_count', 'N/A'))
                        with col2:
                            st.metric(" Duration", f"~{audio_info.get('estimated_duration_minutes', 'N/A')} min")

                    # Display audio player
                    audio_file = Path(audio_data["audio_file"])
                    if audio_file.exists():
                        with open(audio_file, 'rb') as f:
                            audio_bytes = f.read()
                            st.audio(audio_bytes, format='audio/mp3')

                    # Show script
                    script_file = audio_data.get("script_file")
                    if script_file and Path(script_file).exists():
                        st.markdown("** Audio Script:**\n")
                        with open(script_file, 'r') as f:
                            script_content = f.read()
                            with st.expander("Click to view full transcript"):
                                st.text_area("Transcript", script_content, height=300, key="new_script")
                else:
                    st.markdown(response)

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})


def parse_intent(query: str, context: dict) -> dict:
    """Parse user intent using LLM to determine what action to take"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Get context about what exists in the repo
        modules_list = list(context.get("module_summaries", {}).keys())

        # Sample some function names
        sample_functions = []
        for module_funcs in list(context.get("function_summaries", {}).values())[:3]:
            sample_functions.extend(list(module_funcs.keys())[:3])

        prompt = f"""You are an intelligent assistant analyzing user queries about a code repository.

User query: "{query}"

CONTEXT - What exists in this repository:
- Modules: {', '.join(modules_list[:10])}
- Sample functions: {', '.join(sample_functions[:15])}

Your task: Determine what the user wants and extract relevant parameters.

ACTIONS YOU CAN TAKE:

1. get_repo_summary - User wants high-level overview
2. list_modules - User wants to see all files/modules
3. list_all_functions - User wants to see ALL functions across ALL modules
4. get_module_summary - User wants to understand a specific module/file (extract: module_name)
5. search_functions - User asks "how does X work", "find X" about functions (extract: keyword)
6. get_function_summary - User mentions specific function (extract: function_name, optional module_name)
7. list_functions_in_module - User wants functions in specific module (extract: module_name)
8. search_modules - User searches for modules by keyword (extract: keyword)
9. get_stats - User wants statistics about the codebase
10. get_dead_code_summary - User asks about dead code overview/statistics
11. list_dead_code - User wants to see all confirmed dead code
12. search_dead_code - User asks WHY something can be deleted, wants details about specific dead code
    Examples: "why can I delete X", "explain X", "why delete X", "how to delete X", "show details about X"
    Extract: keyword (the name of the function/class/variable they're asking about)
13. show_architecture - User asks about architecture, logical groups, visual diagram
14. show_dependencies - User asks about dependencies, imports, module relationships
15. listen_to_summary - User wants to listen to audio, play audio summary, or hear audio narration
    Examples: "listen to summary", "play audio", "audio explanation", "listen to repo", "play narration"
    Extract: nothing

IMPORTANT:
- If user asks "why can I delete X" or "explain X" or "how can I delete X" where X appears to be a function/class name, use search_dead_code
- If user asks about finding/searching modules or files, use search_modules
- If user asks about functions in a specific file/module, use list_functions_in_module

Respond ONLY with valid JSON:
{{
  "action": "<action_name>",
  "parameters": {{}}
}}"""

        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON response
        response_text = response.content[0].text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        return json.loads(response_text)

    except Exception as e:
        return None


def execute_intent(intent: dict, context: dict) -> str:
    """Execute parsed intent and return formatted response"""
    if not intent:
        return "⚠️ Could not understand the query. Try asking about modules, functions, or dead code."

    action = intent.get("action")
    params = intent.get("parameters", {})

    # Repository Summary
    if action == "get_repo_summary":
        summary = context.get("repo_summary", {})
        if summary and 'human' in summary:
            result = f"** Repository Summary:**\n\n{summary['human']}\n\n"
            if 'technical' in summary:
                result += f"** Technical Summary:**\n\n{summary['technical']}"
            return result
        return "No summary available"

    # List all modules
    elif action == "list_modules":
        modules = context.get("module_summaries", {})
        if not modules:
            return "No modules found"
        result = f"** Modules ({len(modules)} total):**\n\n"
        for module in sorted(modules.keys()):
            result += f" {module}\n"
        return result

    # List ALL functions
    elif action == "list_all_functions":
        function_summaries = context.get("function_summaries", {})
        if not function_summaries:
            return "No functions found"

        result = f"**⚡ All Functions:**\n\n"
        total = 0
        for module_name, functions in sorted(function_summaries.items()):
            if functions:
                result += f"\n** {module_name}** ({len(functions)} functions):\n"
                for func_name in sorted(functions.keys()):
                    result += f"   `{func_name}()`\n"
                    total += 1
        result += f"\n**Total: {total} functions**"
        return result

    # Get module summary
    elif action == "get_module_summary":
        module_name = params.get("module_name", "")
        module_summaries = context.get("module_summaries", {})
        for module_path, summary in module_summaries.items():
            if module_name.lower() in module_path.lower():
                result = f"** {module_path}:**\n\n{summary.get('human', 'No summary')}"
                if 'technical' in summary:
                    result += f"\n\n**Technical Details:**\n{summary['technical']}"
                return result
        return f"Module '{module_name}' not found"

    # Get function summary
    elif action == "get_function_summary":
        function_name = params.get("function_name", "")
        function_summaries = context.get("function_summaries", {})

        for module_name, functions in function_summaries.items():
            if function_name in functions:
                summary = functions[function_name]
                result = f"** `{function_name}()` in {module_name}:**\n\n"
                if isinstance(summary, dict):
                    result += summary.get('human', 'No summary')
                else:
                    result += summary
                return result
        return f"Function '{function_name}' not found"

    # Search functions
    elif action == "search_functions":
        keyword = params.get("keyword", "").lower()
        function_summaries = context.get("function_summaries", {})
        results = []

        for module_name, functions in function_summaries.items():
            for func_name, summary in functions.items():
                if keyword in func_name.lower() or keyword in str(summary).lower():
                    results.append((module_name, func_name, summary))

        if not results:
            return f"No functions found matching '{keyword}'"

        result = f"** Functions matching '{keyword}' ({len(results)} found):**\n\n"
        for module, func, summary in results[:10]:  
            result += f"** `{func}()` in {module}:**\n\n"
            if isinstance(summary, dict):
                result += f"**Human Summary:**\n{summary.get('human', 'No summary')}\n\n"
                if 'technical' in summary:
                    result += f"**Technical Summary:**\n{summary.get('technical')}\n\n"
            else:
                result += f"{str(summary)}\n\n"
            result += "---\n\n"
        return result

    # Get stats
    elif action == "get_stats":
        ast_results = context.get("ast_results", {})
        summary = ast_results.get("__analysis_summary__", {})
        return f"""** Repository Statistics:**

• **Total Modules:** {summary.get('total_modules', 0)}
• **Total Functions:** {summary.get('total_functions', 0)}
• **Total Classes:** {summary.get('total_classes', 0)}
• **Function Summaries:** {len(context.get('function_summaries', {}))}
• **Module Summaries:** {len(context.get('module_summaries', {}))}"""

    # Dead code summary
    elif action == "get_dead_code_summary":
        dead_code = context.get("dead_code_analysis", {})
        if not dead_code:
            return "No dead code analysis available"

        unreferenced_functions = len(dead_code.get("unreferenced_functions", []))
        unused_classes = len(dead_code.get("unused_classes", []))
        unused_imports = len(dead_code.get("unused_imports", []))

        return f"""** Dead Code Summary:**

• **Unreferenced Functions:** {unreferenced_functions}
• **Unused Classes:** {unused_classes}
• **Unused Imports:** {unused_imports}

 Use the Dead Code page for detailed analysis."""

    # List dead code
    elif action == "list_dead_code":
        dead_code = context.get("dead_code_analysis", {})
        if not dead_code:
            return "No dead code analysis available"

        result = "** Dead Code Analysis - What You Can Delete:**\n\n"

        # Unreferenced Functions
        unreferenced_functions = dead_code.get("unreferenced_functions", [])
        if unreferenced_functions:
            result += f"** Unreferenced Functions ({len(unreferenced_functions)}):**\n"
            for func in unreferenced_functions:
                file_name = func.get('module') or func.get('file', 'Unknown')
                result += f" `{func.get('name')}` in {file_name}:{func.get('lineno', '?')}\n"
            result += "\n"

        # Unreferenced Classes
        unused_classes = dead_code.get("unused_classes", [])
        if unused_classes:
            result += f"** Unreferenced Classes ({len(unused_classes)}):**\n"
            for cls in unused_classes:
                file_name = cls.get('module') or cls.get('file', 'Unknown')
                result += f" `{cls.get('name')}` in {file_name}:{cls.get('lineno', '?')}\n"
            result += "\n"

        # Unreferenced Imports
        unused_imports = dead_code.get("unused_imports", [])
        if unused_imports:
            result += f"** Unreferenced Imports ({len(unused_imports)}):**\n"
            for imp in unused_imports:
                file_name = imp.get('module') or imp.get('file', 'Unknown')
                result += f" `{imp.get('name')}` in {file_name}:{imp.get('lineno', '?')}\n"
            result += "\n"

        # Unreferenced Global Variables
        unused_globals = dead_code.get("unused_global_variables", [])
        if unused_globals:
            result += f"** Unreferenced Global Variables ({len(unused_globals)}):**\n"
            for glob in unused_globals:
                file_name = glob.get('module') or glob.get('file', 'Unknown')
                result += f" `{glob.get('name')}` in {file_name}:{glob.get('lineno', '?')}\n"
            result += "\n"

        # Suspicious Patterns
        suspicious_patterns = dead_code.get("suspicious_patterns", [])
        if suspicious_patterns:
            result += f"** Suspicious Patterns ({len(suspicious_patterns)}):**\n"
            for pattern in suspicious_patterns:
                file_name = pattern.get('module') or pattern.get('file', 'Unknown')
                pattern_type = pattern.get('pattern_type', pattern.get('type', 'Unknown'))
                result += f" {pattern_type} - `{pattern.get('name', 'N/A')}` in {file_name}:{pattern.get('lineno', '?')}\n"
            result += "\n"

        total = (len(unreferenced_functions) + len(unused_classes) +
                len(unused_imports) + len(unused_globals) + len(suspicious_patterns))

        if total == 0:
            return " No dead code found! Your codebase looks clean."

        result += f"---\n**Total: {total} items can potentially be removed**\n\n"
        result += " Review each item on the **Dead Code** page for detailed reasons and recommendations."

        return result

    # Search dead code
    elif action == "search_dead_code":
        keyword = params.get("keyword", "").lower()
        dead_code = context.get("dead_code_analysis", {})

        if not dead_code:
            return "No dead code analysis available"

        results = []

        # Search across all dead code categories
        for category in ['unreferenced_functions', 'unused_classes', 'unused_global_variables',
                        'unused_imports', 'suspicious_patterns']:
            items = dead_code.get(category, [])

            for item in items:
                name = item.get('name', item.get('reason', ''))
                module = item.get('module', '')
                reason = item.get('reason', '')

                if (keyword in name.lower() or
                    keyword in module.lower() or
                    keyword in reason.lower()):
                    results.append((category, item))

        if not results:
            return f"No dead code found matching '{keyword}'"

        result = f"** Dead Code Details for '{keyword}':**\n\n"

        for category, item in results:
            name = item.get('name', item.get('reason', 'N/A'))
            file_name = item.get('module') or item.get('file', 'N/A')
            lineno = item.get('lineno', '?')
            status = item.get('status', 'unknown')
            confidence = item.get('confidence', 0)
            reason = item.get('reason', 'N/A')
            recommendation = item.get('recommendation', 'investigate')

            # Get LLM analysis if available
            llm_analysis = item.get('llm_analysis', {})
            evidence = llm_analysis.get('evidence', [])

            result += f"**• `{name}`**\n"
            result += f"   **Category:** {category.replace('_', ' ').title()}\n"
            result += f"   **Location:** {file_name}:{lineno}\n"
            result += f"   **Confidence:** {confidence}%\n"
            result += f"   **Status:** {status}\n"
            result += f"   **Reason:** {reason}\n"
            result += f"   **Recommendation:** {recommendation}\n"

            if evidence:
                result += f"   **Evidence:**\n"
                for ev in evidence:
                    result += f"    - {ev}\n"

            result += "\n---\n\n"

        result += f"**Found {len(results)} item(s)**"
        return result

    # Search modules
    elif action == "search_modules":
        keyword = params.get("keyword", "").lower()
        module_summaries = context.get("module_summaries", {})

        if not module_summaries:
            return "No modules found"

        results = []
        for module_path, summary in module_summaries.items():
            summary_text = str(summary.get('human', '')) + str(summary.get('technical', ''))
            if keyword in module_path.lower() or keyword in summary_text.lower():
                results.append((module_path, summary))

        if not results:
            return f"No modules found matching '{keyword}'"

        result = f"** Modules matching '{keyword}' ({len(results)} found):**\n\n"
        for module, summary in results:
            result += f"** {module}:**\n"
            result += f"{summary.get('human', 'No summary')}\n\n"
            result += "---\n\n"

        return result

    # List functions in module
    elif action == "list_functions_in_module":
        module_name = params.get("module_name", "").lower()

        # Find matching module in AST results
        ast_results = context.get("ast_results", {})

        for module_path, data in ast_results.items():
            if module_path == '__analysis_summary__':
                continue

            if module_name in module_path.lower():
                functions = data.get("functions", [])

                if not functions:
                    return f"No functions found in {module_path}"

                result = f"** Functions in {module_path} ({len(functions)} total):**\n\n"
                for func in functions:
                    func_name = func.get('name', 'Unknown')
                    lineno = func.get('lineno', '?')
                    args = func.get('args', [])
                    arg_names = ', '.join([arg.get('name', '') for arg in args])

                    result += f"• **`{func_name}({arg_names})`** - Line {lineno}\n"

                # Also show summaries if available
                function_summaries = context.get("function_summaries", {}).get(module_path, {})
                if function_summaries:
                    result += f"\n** Function Summaries:**\n\n"
                    for func_name, summary in function_summaries.items():
                        result += f"**`{func_name}()`:**\n"
                        if isinstance(summary, dict):
                            result += f"{summary.get('human', 'No summary')}\n\n"
                        else:
                            result += f"{summary}\n\n"

                return result

        return f"Module containing '{module_name}' not found"

    # Show architecture
    elif action == "show_architecture":
        from pathlib import Path
        import base64
        import json
        import urllib.parse

        repo_name = context.get("repo_name", "Unknown")
        mermaid_file = Path(f"diagrams/mermaid_{repo_name}.mmd")

        if mermaid_file.exists():
            with open(mermaid_file, 'r') as f:
                mermaid_code = f.read()

            # Create mermaid.live link with pre-loaded diagram
            # Format: https://mermaid.live/edit#base64:{encoded_state}
            state = {
                "code": mermaid_code,
                "mermaid": {"theme": "default"},
                "autoSync": True,
                "updateDiagram": True
            }
            state_json = json.dumps(state)
            encoded_state = base64.b64encode(state_json.encode('utf-8')).decode('utf-8')
            mermaid_link = f"https://mermaid.live/edit#base64:{encoded_state}"

            result = f"** Visual Architecture Diagram (Mermaid.js):**\n\n"
            result += f" **[Click here to view interactive diagram in Mermaid Live]({mermaid_link})**\n\n"
            result += f"```mermaid\n{mermaid_code}\n```\n\n"

            # Also show logical groups
            logical_groups = context.get("logical_groups", {})
            if logical_groups:
                result += f"** Logical Groups:**\n\n"
                for group_name, files in logical_groups.items():
                    result += f"**{group_name}** ({len(files)} files):\n"
                    for file in sorted(files):
                        result += f"  • {file}\n"
                    result += "\n"

            return result
        else:
            return "⚠️ Architecture diagram not found. Please check the Architecture page."

    # Show dependencies
    elif action == "show_dependencies":
        from pathlib import Path
        repo_name = context.get("repo_name", "Unknown")
        deps_file = Path(f"diagrams/dependencies_{repo_name}.txt")

        if deps_file.exists():
            with open(deps_file, 'r') as f:
                deps_content = f.read()

            result = f"** Module Dependencies:**\n\n```\n{deps_content}\n```"
            return result
        else:
            return "⚠️ Dependency graph not found. Please check the Dependencies page."

    # Listen to summary (audio)
    elif action == "listen_to_summary":
        from pathlib import Path
        repo_name = context.get("repo_name", "Unknown")
        audio_file = Path(f"audio/{repo_name}_summary.mp3")
        script_file = Path(f"audio/{repo_name}_audio_script.txt")

        if not audio_file.exists():
            return "⚠️ No audio narration found. Please check the **Audio** page."

        # Return special marker to trigger audio player in chat
        return {
            "type": "audio",
            "audio_file": str(audio_file),
            "script_file": str(script_file) if script_file.exists() else None,
            "audio_info": context.get("audio_narration", {})
        }

    return "⚠️ Action not supported yet"


def generate_chat_response(query: str, context: dict):
    """Generate response to user query using intent detection (like CLI)"""
    try:
        # Parse intent
        intent = parse_intent(query, context)

        # Execute intent and get response
        response = execute_intent(intent, context)
        return response

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


def main():
    """Main Streamlit app"""

    # Get repo name from command line argument 
    import sys
    default_repo = sys.argv[1] if len(sys.argv) > 1 else None

    # Sidebar
    with st.sidebar:
        st.title(" RepoSpeak")
        st.markdown("---")

        # Repository selection
        contexts = load_context_files()

        if not contexts:
            st.error("No context files found!")
            st.info("Run: `python3 repospeak.py <repo_path>`")
            st.stop()

        # Determine default index
        repo_list = list(contexts.keys())
        default_index = 0
        if default_repo and default_repo in repo_list:
            default_index = repo_list.index(default_repo)

        selected_repo = st.selectbox(
            "Select Repository",
            options=repo_list,
            index=default_index,
            key="repo_selector"
        )

        context_file = contexts[selected_repo]
        context = load_context(context_file)

        st.markdown("---")

        # Navigation
        st.subheader(" Navigation")
        page = st.radio(
            "Go to:",
            options=[
                "💬 Chat",
                "📊 Overview",
                "🎧 Audio",
                "🎨 Architecture",
                "🔗 Dependencies",
                "💀 Dead Code",
                "⚡ Functions",
                "📚 Modules"
            ],
            key="navigation"
        )

        st.markdown("---")
        st.caption(f"Last analyzed: {datetime.now().strftime('%Y-%m-%d')}")

    # Main content area
    if page == "💬 Chat":
        chat_interface(context)
    elif page == "📊 Overview":
        display_overview(context)
    elif page == "🎧 Audio":
        display_audio(context)
    elif page == "🎨 Architecture":
        display_architecture(context)
    elif page == "🔗 Dependencies":
        display_dependencies(context)
    elif page == "💀 Dead Code":
        display_dead_code(context)
    elif page == "⚡ Functions":
        display_functions(context)
    elif page == "📚 Modules":
        display_modules(context)


if __name__ == "__main__":
    main()
