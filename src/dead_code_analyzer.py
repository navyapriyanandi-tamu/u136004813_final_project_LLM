#!/usr/bin/env python3
"""
LLM-Powered Dead Code Analyzer
Analyzes AST dead code candidates using Claude to distinguish genuine dead code from false positives
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import anthropic

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANTHROPIC_API_KEY


class DeadCodeAnalyzer:
    """
    Analyzes dead code candidates using LLM reasoning

    Takes AST-detected dead code candidates and uses Claude to:
    1. Distinguish genuine dead code from false positives
    2. Provide confidence scores
    3. Give detailed explanations
    4. Recommend actions
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = "claude-sonnet-4-20250514"

    def analyze_dead_code(self, ast_results: Dict) -> Dict:
        """
        Analyze all dead code candidates from AST results
        """
        summary = ast_results.get('__analysis_summary__', {})
        dead_code = summary.get('dead_code_candidates', {})

        print("\nStarting LLM Dead Code Analysis")

        results = {
            'unreferenced_functions': [],
            'unused_classes': [],
            'unused_imports': [],
            'unused_global_variables': [],
            'unreachable_code': [],
            'suspicious_patterns': [],
            'summary': {
                'total_analyzed': 0,
                'confirmed_dead_code': 0,
                'false_positives': 0,
                'uncertain': 0,
                'avg_confidence': 0.0
            }
        }

        # Analyze each category
        categories = [
            ('unreferenced_functions', 'function'),
            ('unused_classes', 'class'),
            ('unused_global_variables', 'variable'),
            ('unreachable_code', 'unreachable_code'),
            ('suspicious_patterns', 'pattern')
        ]

        for category, item_type in categories:
            candidates = dead_code.get(category, [])

            # Filter only items that need LLM analysis
            needs_llm = [c for c in candidates if c.get('needs_llm', True)]

            if not needs_llm:
                continue

            print(f"\n Analyzing {len(needs_llm)} {category}")

            # Batch analyze (5 items per API call to save tokens)
            for i in range(0, len(needs_llm), 5):
                batch = needs_llm[i:i+5]

                # Gather context for each item in batch
                batch_with_context = []
                for candidate in batch:
                    context = self._gather_context(candidate, ast_results, item_type)
                    batch_with_context.append(context)

                # Analyze batch with LLM
                analyses = self._analyze_batch(batch_with_context, item_type)

                # Store results
                for analysis in analyses:
                    results[category].append(analysis)
                    results['summary']['total_analyzed'] += 1

                    if analysis['status'] == 'dead_code':
                        results['summary']['confirmed_dead_code'] += 1
                    elif analysis['status'] == 'false_positive':
                        results['summary']['false_positives'] += 1
                    else:
                        results['summary']['uncertain'] += 1

                print(f"   Processed {min(i+5, len(needs_llm))}/{len(needs_llm)}")

        # Calculate average confidence
        if results['summary']['total_analyzed'] > 0:
            total_confidence = sum(
                item.get('confidence', 0)
                for category in categories
                for item in results[category[0]]
            )
            results['summary']['avg_confidence'] = total_confidence / results['summary']['total_analyzed']

        print(f"\n Analysis complete!")
        print(f"   Total analyzed: {results['summary']['total_analyzed']}")
        print(f"   Confirmed dead code: {results['summary']['confirmed_dead_code']}")
        print(f"   False positives: {results['summary']['false_positives']}")
        print(f"   Uncertain: {results['summary']['uncertain']}")
        print(f"   Average confidence: {results['summary']['avg_confidence']:.1f}%")

        return results

    def _gather_context(self, candidate: Dict, ast_results: Dict, item_type: str) -> Dict:
        """
        Gather rich context for a dead code candidate
        """
        context = {
            'candidate': candidate,
            'type': item_type,
            'code': None,
            'docstring': None,
            'class_context': None,
            'module_context': None,
            'project_context': None
        }

        module_path = candidate.get('module', '')
        module_data = ast_results.get(module_path, {})

        # Get project-level context
        summary = ast_results.get('__analysis_summary__', {})
        context['project_context'] = {
            'total_modules': summary.get('total_modules', 0),
            'total_functions': summary.get('total_functions', 0),
            'is_library': self._is_library_project(ast_results)
        }

        # Get module context
        context['module_context'] = {
            'path': module_path,
            'docstring': module_data.get('module_docstring', ''),
            'is_init': module_path.endswith('__init__.py'),
            'is_config': any(module_path.endswith(cf) for cf in ['conf.py', 'setup.py', 'conftest.py'])
        }

        # Type-specific context
        if item_type == 'function':
            context.update(self._get_function_context(candidate, module_data, ast_results))
        elif item_type == 'class':
            context.update(self._get_class_context(candidate, module_data, ast_results))
        elif item_type == 'variable':
            context.update(self._get_variable_context(candidate, module_data))

        return context

    def _get_function_context(self, candidate: Dict, module_data: Dict, ast_results: Dict) -> Dict:
        """Get context specific to functions"""
        func_name = candidate['name']

        # Find the function in module data
        functions = module_data.get('functions', [])
        func_data = next((f for f in functions if f['name'] == func_name), None)

        context = {
            'code': func_data.get('code', '') if func_data else '',
            'docstring': func_data.get('docstring', '') if func_data else '',
            'decorators': func_data.get('decorators', []) if func_data else [],
            'is_method': False,
            'class_context': None
        }

        # Check if it's a class method
        for cls in module_data.get('classes', []):
            if func_name in cls.get('methods', []):
                context['is_method'] = True
                context['class_context'] = {
                    'name': cls['name'],
                    'base_classes': cls.get('base_classes', []),
                    'docstring': cls.get('docstring', ''),
                    'methods': cls.get('methods', []),
                    'is_exported': self._is_exported(cls['name'], ast_results)
                }
                break

        return context

    def _get_class_context(self, candidate: Dict, module_data: Dict, ast_results: Dict) -> Dict:
        """Get context specific to classes"""
        cls_name = candidate['name']

        # Find the class in module data
        classes = module_data.get('classes', [])
        cls_data = next((c for c in classes if c['name'] == cls_name), None)

        return {
            'code': f"class {cls_name}({', '.join(cls_data.get('base_classes', []))}):" if cls_data else '',
            'docstring': cls_data.get('docstring', '') if cls_data else '',
            'base_classes': cls_data.get('base_classes', []) if cls_data else [],
            'methods': cls_data.get('methods', []) if cls_data else []
        }

    def _get_variable_context(self, candidate: Dict, module_data: Dict) -> Dict:
        """Get context specific to variables"""
        return {
            'assigned_to': candidate.get('assigned_to', 'unknown'),
            'module_docstring': module_data.get('module_docstring', '')
        }

    def _is_library_project(self, ast_results: Dict) -> bool:
        """
        Detect if this is a library or application

        Libraries: Provide reusable functionality for other developers
        Applications: Standalone programs with entry points (main, CLI, etc.)
        """
        application_indicators = 0
        library_indicators = 0

        main_count = 0
        init_with_exports_count = 0

        for module_path, data in ast_results.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            # Check for application indicators
           
            functions = data.get('functions', [])
            has_main_here = any(f.get('name') == 'main' for f in functions)
            if has_main_here:
                main_count += 1
                # Only count as application if in main.py/app.py/cli.py at root
                if any(module_path.endswith(name) for name in ['main.py', 'app.py', 'cli.py', 'run.py']):
                    if '/' not in module_path or module_path.count('/') <= 1:
                        application_indicators += 5  

            # 2. Has CLI argument parsing (argparse, click, etc.)
            has_cli_parsing = any(imp.get('module', '') in ['argparse', 'click', 'fire', 'typer']
                   for imp in data.get('imports', []))
            if has_cli_parsing:
                application_indicators += 3  
                # If main() + CLI parsing in same file = definitely application
                if has_main_here:
                    application_indicators += 5  

            # Check for library indicators
           
            if module_path.endswith('__init__.py'):
                imports = data.get('imports', [])
                # Relative imports: module starts with '.' OR module is empty (from . import X)
                from_imports = [imp for imp in imports
                               if imp.get('type') == 'from_import' and
                               (imp.get('module', '').startswith('.') or imp.get('module', '') == '')]
                if len(from_imports) > 0:
                    init_with_exports_count += 1
                    library_indicators += 2  

                # Check for __all__ in variables 
                variables = data.get('variables', [])
                if any(v.get('name') == '__all__' for v in variables):
                    library_indicators += 3  

        # Multiple __init__.py with exports = definitely a library
        if init_with_exports_count >= 2:
            library_indicators += 5  

        # Single utility main() in help.py/version.py = library utility
        if main_count == 1 and library_indicators > 0:
            application_indicators = max(0, application_indicators - 3)

        # If more application indicators, it's an application
        is_library = library_indicators > application_indicators

        # Debug output
        print(f"   Project type detection: application={application_indicators}, library={library_indicators}")
        print(f"      (main_count={main_count}, init_with_exports={init_with_exports_count})")
        print(f"   -> Classified as: {'LIBRARY' if is_library else 'APPLICATION'}")

        return is_library

    def _is_exported(self, name: str, ast_results: Dict) -> bool:
        """Check if a symbol is exported in __init__.py"""
        for module_path, data in ast_results.items():
            if module_path.endswith('__init__.py'):
                # Check imports
                for imp in data.get('imports', []):
                    if imp.get('type') == 'from_import' and imp.get('name') == name:
                        return True
        return False

    def _analyze_batch(self, batch_with_context: List[Dict], item_type: str) -> List[Dict]:
        """
        Analyze a batch of candidates with LLM

        """
        # Build prompt
        prompt = self._build_batch_prompt(batch_with_context, item_type)

        # Call Claude
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse response
            response_text = response.content[0].text

            # Extract JSON from response
            analyses = self._parse_llm_response(response_text, batch_with_context)

            return analyses

        except Exception as e:
            print(f"   ⚠️  LLM API error: {e}")
            # Return default uncertain results
            return [{
                **item['candidate'],
                'status': 'uncertain',
                'confidence': 0,
                'reason': f'LLM analysis failed: {str(e)}',
                'recommendation': 'investigate',
                'evidence': []
            } for item in batch_with_context]

    def _build_batch_prompt(self, batch: List[Dict], item_type: str) -> str:
        """Build comprehensive prompt for batch analysis"""

        # Get project context from first item
        project_type = "library" if batch[0]['project_context']['is_library'] else "application"

        prompt = f"""You are analyzing potential dead code in a Python {project_type} project.

{'='*80}
IMPORTANT CONTEXT:
{'='*80}

This is a **{project_type.upper()}** project. This matters A LOT:

**If APPLICATION:**
- ALL code should be called somewhere in the codebase
- Even if a class is "exported" in __init__.py, it's just for code organization
- If a method is NEVER called internally, it's DEAD CODE (not public API)
- Be AGGRESSIVE about flagging unused methods as dead_code in applications
- Only mark as false_positive if there's STRONG evidence (decorators, hooks, callbacks, etc.)

**If LIBRARY:**
- Methods of exported classes MAY be public API for external users
- If class is exported AND method is well-documented, likely public API
- Be more conservative, mark as uncertain or false_positive if exported

{'='*80}
ANALYZE THESE {len(batch)} ITEMS:
{'='*80}

"""

        # Add each item with context
        for i, item in enumerate(batch, 1):
            candidate = item['candidate']

            # Get name (or description for unreachable code)
            item_name = candidate.get('name', candidate.get('reason', f"{item_type} at line {candidate.get('lineno')}"))

            prompt += f"""
{'-'*80}
ITEM #{i}: {item_name} ({item_type})
{'-'*80}

**Location:** {candidate.get('module')}:{candidate.get('lineno')}

"""

            # Add code
            if item.get('code'):
                code_preview = item['code'][:500]  # Limit to 500 chars
                prompt += f"""**Code:**
```python
{code_preview}{'...' if len(item['code']) > 500 else ''}
```

"""

            # Add docstring
            if item.get('docstring'):
                prompt += f"""**Docstring:** {item['docstring'][:200]}

"""

            # Add class context for methods
            if item.get('is_method') and item.get('class_context'):
                cls_ctx = item['class_context']
                prompt += f"""**Class Context:**
- Parent class: {cls_ctx['name']}
- Base classes: {cls_ctx['base_classes']}
- Class is EXPORTED: {cls_ctx['is_exported']}
- Other methods: {', '.join(cls_ctx['methods'][:5])}{'...' if len(cls_ctx['methods']) > 5 else ''}

"""

            # Add decorators
            if item.get('decorators'):
                prompt += f"""**Decorators:** {[d.get('name') for d in item['decorators']]}

"""

            # Add module context
            if item['module_context'].get('is_init'):
                prompt += f"""**NOTE:** This is in __init__.py (package initialization file)

"""

            if item['module_context'].get('is_config'):
                prompt += f"""**NOTE:** This is in a config file (likely used by external tools)

"""

        # Add analysis instructions
        prompt += f"""
{'='*80}
YOUR TASK:
{'='*80}

For EACH item above, determine if it's genuinely dead code or a false positive.

**Consider:**

1. **Public API Methods** (for libraries):
   - If parent class is exported in __init__.py, its methods are public API
   - Users call these externally, so no internal calls is NORMAL
   - Status: false_positive

2. **Interface/Protocol Methods**:
   - Methods like get, keys, __iter__, __len__ implement protocols
   - Called by Python/frameworks, not directly by code
   - Status: false_positive

3. **Callback/Hook Methods**:
   - Methods with names like handle_*, on_*, process_*
   - Passed to frameworks as callbacks
   - Status: false_positive

4. **Deprecated Code**:
   - Docstring contains "deprecated", "use X instead"
   - Status: dead_code (safe to remove)

5. **Config File Variables**:
   - In conf.py, setup.py → used by tools like Sphinx, setuptools
   - Status: false_positive

6. **Genuinely Unused**:
   - No clear purpose, no external usage pattern
   - Status: dead_code

**Output Format:**

Return a JSON array with one object per item (in the same order as listed above):

```json
[
  {{
    "name": "item_name_or_description",
    "status": "dead_code" | "false_positive" | "uncertain",
    "confidence": 0-100,
    "category": "deprecated" | "public_api" | "interface_method" | "callback" | "config" | "genuinely_unused" | "unreachable",
    "reason": "Clear 2-3 sentence explanation with specific evidence",
    "recommendation": "safe_to_delete" | "keep" | "mark_deprecated" | "investigate",
    "evidence": [
      "Evidence point 1",
      "Evidence point 2"
    ]
  }}
]
```

**Note:** For unreachable code blocks, use the reason/description as the name field.

**Be conservative:** When in doubt, mark as "false_positive" or "uncertain".
Deleting used code breaks things; keeping dead code is just clutter.

**Return ONLY the JSON array, no other text.**
"""

        return prompt

    def _parse_llm_response(self, response_text: str, batch: List[Dict]) -> List[Dict]:
        """Parse LLM JSON response and merge with candidate data"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0].strip()

            analyses = json.loads(json_text)

            # Merge with original candidate data
            results = []
            for i, analysis in enumerate(analyses):
                if i < len(batch):
                    result = {
                        **batch[i]['candidate'],
                        'llm_analysis': {
                            'status': analysis.get('status', 'uncertain'),
                            'confidence': analysis.get('confidence', 0),
                            'category': analysis.get('category', 'unknown'),
                            'reason': analysis.get('reason', ''),
                            'recommendation': analysis.get('recommendation', 'investigate'),
                            'evidence': analysis.get('evidence', [])
                        },
                        # Flatten for convenience
                        'status': analysis.get('status', 'uncertain'),
                        'confidence': analysis.get('confidence', 0),
                        'reason': analysis.get('reason', ''),
                        'recommendation': analysis.get('recommendation', 'investigate')
                    }
                    results.append(result)

            return results

        except json.JSONDecodeError as e:
            print(f"   ⚠️  Failed to parse LLM response as JSON: {e}")
            print(f"   Response: {response_text[:200]}")

            # Return uncertain results
            return [{
                **item['candidate'],
                'status': 'uncertain',
                'confidence': 0,
                'reason': 'Failed to parse LLM response',
                'recommendation': 'investigate',
                'evidence': []
            } for item in batch]


def analyze_dead_code_with_llm(ast_results_file: str, output_file: str = None):
    """
    Analyze dead code from AST results using LLM

    """
    # Load AST results
    with open(ast_results_file, 'r') as f:
        ast_results = json.load(f)

    # Analyze with LLM
    analyzer = DeadCodeAnalyzer()
    results = analyzer.analyze_dead_code(ast_results)

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dead_code_analyzer.py <ast_results.json> [output.json]")
        sys.exit(1)

    ast_file = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    analyze_dead_code_with_llm(ast_file, output)
