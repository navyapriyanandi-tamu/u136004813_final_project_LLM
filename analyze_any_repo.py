#!/usr/bin/env python3
"""
Generic AST Parser - Analyze ANY Python Repository

Usage:
    python analyze_any_repo.py <path_to_repo>

Example:
    python analyze_any_repo.py ../test_project
    python analyze_any_repo.py /path/to/django/project
    python analyze_any_repo.py .  (analyze current directory)
"""

import sys
import json
from pathlib import Path
from src.ast_parser import CodeAnalyzer
from src.function_summarizer import FunctionSummarizer
from src.module_summarizer import ModuleSummarizer
from src.repo_summarizer import RepoSummarizer
from src.context_manager import ContextManager
from src.dead_code_analyzer import DeadCodeAnalyzer
from src.module_grouper import ModuleGrouper
from src.diagram_generator import DiagramGenerator


def analyze_repository(repo_path: str):
    """Analyze any Python repository and output its structure"""

    repo_path_obj = Path(repo_path)

    if not repo_path_obj.exists():
        print(f"❌ Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"ANALYZING REPOSITORY: {repo_path_obj.resolve()}")
    print("=" * 80)

    # Check if context file already exists
    context_file = f"context_{repo_path_obj.name}.json"
    context = ContextManager(repo_path_obj.name)

    if Path(context_file).exists():
        print(f"\n Loading existing context: {context_file}")
        context.load_from_file(context_file)

    # Run AST analysis 
    print(f"\n Running AST analysis")
    analyzer = CodeAnalyzer(repo_path)
    results = analyzer.analyze_repository()

    # Store AST results in context
    context.store_ast_results(results)

    # Get summary
    summary = results.get('__analysis_summary__', {})

    print("\n" + "=" * 80)
    print("REPOSITORY SUMMARY")
    print("=" * 80)

    print(f"\n Statistics:")
    print(f"   Total modules: {summary.get('total_modules', 0)}")
    print(f"   Total functions: {summary.get('total_functions', 0)}")
    print(f"   Total classes: {summary.get('total_classes', 0)}")

    # Dead code
    dead_code = summary.get('dead_code_candidates', {})
    unreferenced_funcs = dead_code.get('unreferenced_functions', [])
    unused_classes = dead_code.get('unused_classes', [])

    print(f"\n Dead Code Candidates:")
    print(f"   Unreferenced functions: {len(unreferenced_funcs)}")
    print(f"   Unused classes: {len(unused_classes)}")

    # Module list
    modules = [k for k in results.keys() if k != '__analysis_summary__']
    print(f"\n Modules ({len(modules)}):")
    for i, mod in enumerate(modules[:10], 1):
        print(f"   {i}. {mod}")
    if len(modules) > 10:
        print(f"   ... and {len(modules) - 10} more")

    # Entry points
    exec_flows = summary.get('execution_flows', {})
    if exec_flows:
        print(f"\n Entry Points:")
        for entry_point, flow in exec_flows.items():
            print(f"\n    {entry_point}:")
            for step in flow[:5]:
                print(f"      {step}")
            if len(flow) > 5:
                print(f"      ... and {len(flow) - 5} more steps")

    # Tech stack detection
    print(f"\n  Tech Stack Analysis:")
    tech_stacks = set()
    all_third_party = {}

    for mod_path, mod_data in results.items():
        if mod_path == '__analysis_summary__':
            continue

        dep_class = mod_data.get('dependency_classification', {})
        tech_stacks.update(dep_class.get('tech_stack', []))

        # Collect all third-party packages
        third_party = dep_class.get('third_party', {})
        for category, packages in third_party.items():
            if category not in all_third_party:
                all_third_party[category] = set()
            all_third_party[category].update(packages)

    if tech_stacks:
        print(f"   Tech Stack: {', '.join(sorted(tech_stacks))}")
    else:
        print(f"   Tech Stack: Pure Python (stdlib only)")

    if all_third_party:
        print(f"\n   Third-Party Packages:")
        for category, packages in sorted(all_third_party.items()):
            print(f"      {category.capitalize()}: {', '.join(sorted(packages))}")

    # Module dependencies
    module_deps = summary.get('module_dependencies', {})
    if module_deps:
        print(f"\n Module Dependencies (top 5):")
        for i, (mod, deps) in enumerate(list(module_deps.items())[:5], 1):
            depends_on = deps.get('depends_on', [])
            print(f"   {i}. {mod}")
            print(f"      -> depends on: {', '.join(depends_on[:3])}")
            if len(depends_on) > 3:
                print(f"      ... and {len(depends_on) - 3} more")

    # Dead code details (AST-level detection)
    if unreferenced_funcs:
        print(f"\n⚠️  Unreferenced Functions (AST detection - first 10):")
        for func in unreferenced_funcs[:10]:
            print(f"    {func['name']} in {func['module']}:{func['lineno']}")
        print(f"   Note: Will be analyzed by LLM to distinguish false positives")

    if unused_classes:
        print(f"\n⚠️  Unused Classes (AST detection):")
        for cls in unused_classes[:10]:
            print(f"    {cls['name']} in {cls['module']}:{cls['lineno']}")
        print(f"   Note: Will be analyzed by LLM to distinguish false positives")

    # Generate All Summaries Automatically (Function -> Module -> Repo -> Dead Code)
    print(f"\n Step 1/4: Generating function summaries")
    function_summarizer = FunctionSummarizer()

    for module_path, module_data in results.items():
        if module_path == '__analysis_summary__' or 'error' in module_data:
            continue
        functions = module_data.get('functions', [])
        if functions:
            summaries = function_summarizer.summarize_module_functions(module_path, functions)
            if summaries:
                context.store_function_summaries(module_path, summaries)

    print(f"\n Step 2/4: Generating module summaries")
    module_summarizer = ModuleSummarizer()

    all_function_summaries = context.get_all_function_summaries()
    module_summaries = module_summarizer.summarize_all_modules(results, all_function_summaries)

    for module_path, module_summary in module_summaries.items():
        context.store_module_summary(module_path, module_summary)

    print(f"\n  Step 3/4: Generating repository summary")
    repo_summarizer = RepoSummarizer()

    all_module_summaries = context.get_all_module_summaries()
    repo_summary = repo_summarizer.summarize_repository(
        repo_path_obj.name,
        all_module_summaries,
        summary
    )

    context.store_repo_summary(repo_summary)

    # Step 4: LLM-powered dead code analysis
    print(f"\n Step 4/6: Analyzing dead code with LLM")
    dead_code_analyzer = DeadCodeAnalyzer()
    dead_code_analysis = dead_code_analyzer.analyze_dead_code(results)

    # Store dead code analysis in context
    context.store_dead_code_analysis(dead_code_analysis)

    # Step 5: Generate architecture diagrams
    print(f"\n Step 5/6: Generating architecture diagrams")
    module_grouper = ModuleGrouper()
    all_module_summaries = context.get_all_module_summaries()
    logical_groups = module_grouper.group_modules(all_module_summaries)

    # Store logical groups in context
    context.context["logical_groups"] = logical_groups

    # Generate diagrams
    diagram_gen = DiagramGenerator(repo_path_obj.name)
    diagram_gen.generate_all_diagrams(logical_groups, results)

    # Step 6: Generate audio narration
    print(f"\n  Step 6/6: Generating audio narration")
    from src.audio_generator import AudioGenerator

    audio_gen = AudioGenerator(repo_path_obj.name)
    repo_summary_data = context.get_repo_summary()

    if repo_summary_data and 'human' in repo_summary_data:
        try:
            audio_result = audio_gen.generate_repository_audio(repo_summary_data['human'])
            print(f"    Audio saved: {audio_result['audio_file']}")
            print(f"    Script saved: {audio_result['script_file']}")
            print(f"    Duration: ~{audio_result['estimated_duration_minutes']} minutes")

            # Store audio info in context
            context.context["audio_narration"] = audio_result
        except Exception as e:
            print(f"   ⚠️  Audio generation failed: {str(e)}")
            print(f"    Continuing without audio")
    else:
        print(f"   ⚠️  No repository summary found, skipping audio generation")

    # Save outputs - 3 consolidated files

    # File 1: AST Analysis (pure parsing output)
    ast_file = f"ast_analysis_{repo_path_obj.name}.json"
    with open(ast_file, 'w') as f:
        json.dump(results, f, indent=2)

    # File 2: All Summaries (function + module + repo)
    summaries_file = f"summaries_{repo_path_obj.name}.json"
    all_summaries_combined = {
        "function_summaries": context.get_all_function_summaries(),
        "module_summaries": context.get_all_module_summaries(),
        "repo_summary": context.get_repo_summary()
    }
    with open(summaries_file, 'w') as f:
        json.dump(all_summaries_combined, f, indent=2)

    # File 3: Dead Code Analysis
    dead_code_file = f"dead_code_analysis_{repo_path_obj.name}.json"
    with open(dead_code_file, 'w') as f:
        json.dump(dead_code_analysis, f, indent=2)

    # Save complete context 
    context.save_to_file(context_file)

    print(f"\n\n" + "=" * 80)
    print("OUTPUT FILES SAVED (3 MAIN FILES + CONTEXT + DIAGRAMS)")
    print("=" * 80)

    print(f"\n File 1: AST Analysis - {ast_file}")
    print(f"   Size: {Path(ast_file).stat().st_size:,} bytes")
    print(f"   Contains:")
    print(f"      • Complete AST parsing results")
    print(f"      • Module structures, functions, classes")
    print(f"      • Import dependencies")
    print(f"      • Dead code candidates (AST-level)")

    print(f"\n File 2: All Summaries - {summaries_file}")
    print(f"   Size: {Path(summaries_file).stat().st_size:,} bytes")
    print(f"   Contains:")
    print(f"      • Function summaries (human + technical)")
    print(f"      • Module summaries (human + technical)")
    print(f"      • Repository summary (high-level overview)")

    print(f"\n File 3: Dead Code Analysis - {dead_code_file}")
    print(f"   Size: {Path(dead_code_file).stat().st_size:,} bytes")
    print(f"   Contains:")
    print(f"      • LLM-analyzed dead code candidates")
    print(f"      • Confidence scores + explanations")
    print(f"      • Recommendations (safe_to_delete/keep/investigate)")

    print(f"\n Context File: {context_file}")
    print(f"   Size: {Path(context_file).stat().st_size:,} bytes")
    print(f"   Contains: Everything combined (AST + summaries + dead code + logical groups)")
    print(f"   Used by: MCP server for Claude Desktop queries")

    print(f"\n Architecture Diagrams: diagrams/")
    print(f"   • architecture_{repo_path_obj.name}.txt - High-level logical architecture")
    print(f"   • dependencies_{repo_path_obj.name}.txt - Module dependency graph")
    print(f"   View with: cat diagrams/architecture_{repo_path_obj.name}.txt")

    print("\n" + "=" * 80)
    print("SAMPLE MODULE STRUCTURE (first module)")
    print("=" * 80)

    # Show one complete module as example
    if modules:
        sample_module = modules[0]
        module_data = results[sample_module]

        print(f"\nModule: {sample_module}")
        print(f"Docstring: {module_data.get('module_docstring', 'None')}")
        print(f"Line count: {module_data.get('line_count', 0)}")
        print(f"Entry point: {module_data.get('is_entry_point', False)}")

        # Functions
        functions = module_data.get('functions', [])
        if functions:
            print(f"\nFunctions ({len(functions)}):")
            for func in functions[:3]:
                args_str = ', '.join([
                    f"{a['name']}: {a.get('type', 'Any')}"
                    for a in func.get('args', [])
                ])
                return_type = func.get('return_type', 'None')
                print(f"   • {func['name']}({args_str}) -> {return_type}")
                if func.get('docstring'):
                    print(f"     \"{func['docstring'][:60]}\"")
                if func.get('decorators'):
                    for dec in func['decorators']:
                        print(f"     @{dec['name']}")

        # Classes
        classes = module_data.get('classes', [])
        if classes:
            print(f"\nClasses ({len(classes)}):")
            for cls in classes[:3]:
                methods = cls.get('methods', [])
                print(f"   • {cls['name']}")
                print(f"     Methods: {', '.join(methods[:5])}")

        # Dependencies
        dep_class = module_data.get('dependency_classification', {})
        if dep_class:
            print(f"\nDependencies:")
            print(f"   Stdlib: {', '.join(dep_class.get('stdlib', [])[:5])}")
            third_party = dep_class.get('third_party', {})
            if third_party:
                for cat, pkgs in third_party.items():
                    print(f"   {cat.capitalize()}: {', '.join(pkgs[:3])}")

    # Display function summaries
    all_summaries = context.get_all_function_summaries()
    if all_summaries:
        print("\n" + "=" * 80)
        print("FUNCTION SUMMARIES")
        print("=" * 80)

        for module_path, summaries in all_summaries.items():
            print(f"\n {module_path}:")
            for func_name, summary_dict in summaries.items():
                print(f"\n   {func_name}:")
                print(f"     Simple: {summary_dict.get('human', 'N/A')}")
                print(f"     Technical: {summary_dict.get('technical', 'N/A')}")

        # Show context stats
        stats = context.get_summary_stats()
        print(f"\n Context Stats:")
        print(f"   Total modules: {stats['total_modules']}")
        print(f"   Function summaries: {stats['total_function_summaries']}")
        print(f"   Module summaries: {stats['modules_with_summaries']}")
        print(f"   Ready for repo summarization")

    # Display module summaries
    all_module_summaries = context.get_all_module_summaries()
    if all_module_summaries:
        print("\n" + "=" * 80)
        print("MODULE SUMMARIES")
        print("=" * 80)

        for module_path, summary in all_module_summaries.items():
            print(f"\n {module_path}:")
            print(f"\n   Human:")
            print(f"     {summary.get('human', 'N/A')}")
            print(f"\n   Technical:")
            print(f"     {summary.get('technical', 'N/A')}")

    # Display repository summary
    repo_summary = context.get_repo_summary()
    if repo_summary:
        print("\n" + "=" * 80)
        print("REPOSITORY SUMMARY")
        print("=" * 80)

        print(f"\n  {context.repo_name}")
        print(f"\n   Human:")
        print(f"     {repo_summary.get('human', 'N/A')}")
        print(f"\n   Technical:")
        print(f"     {repo_summary.get('technical', 'N/A')}")

    # Display dead code analysis summary
    if dead_code_analysis:
        print("\n" + "=" * 80)
        print("DEAD CODE ANALYSIS (LLM-POWERED)")
        print("=" * 80)

        dc_summary = dead_code_analysis.get('summary', {})
        print(f"\n Analysis Summary:")
        print(f"   Total analyzed: {dc_summary.get('total_analyzed', 0)}")
        print(f"   Confirmed dead code: {dc_summary.get('confirmed_dead_code', 0)}")
        print(f"   False positives: {dc_summary.get('false_positives', 0)}")
        print(f"   Uncertain: {dc_summary.get('uncertain', 0)}")
        print(f"   Average confidence: {dc_summary.get('avg_confidence', 0):.1f}%")

        # Show confirmed dead code items
        confirmed_count = dc_summary.get('confirmed_dead_code', 0)
        if confirmed_count > 0:
            print(f"\n Confirmed Dead Code ({confirmed_count} items):")

            for category in ['unreferenced_functions', 'unused_classes', 'unused_global_variables',
                           'unreachable_code', 'suspicious_patterns']:
                items = dead_code_analysis.get(category, [])
                dead_items = [item for item in items if item.get('status') == 'dead_code']

                if dead_items:
                    print(f"\n   {category.replace('_', ' ').title()}:")
                    for item in dead_items[:5]:
                        name = item.get('name', item.get('reason', 'N/A'))
                        location = f"{item.get('module')}:{item.get('lineno')}"
                        confidence = item.get('confidence', 0)
                        recommendation = item.get('recommendation', 'investigate')
                        print(f"         {name}")
                        print(f"         {location}")
                        print(f"         Confidence: {confidence}%")
                        print(f"         {item.get('reason', 'N/A')[:100]}...")
                        print(f"         Recommendation: {recommendation}")

                    if len(dead_items) > 5:
                        print(f"      ... and {len(dead_items) - 5} more")

        print(f"\n  See {dead_code_file} for full analysis with evidence and explanations")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Missing repository path")
        print("\nUsage:")
        print("   python analyze_any_repo.py <path_to_repo>")
        print("\nExamples:")
        print("   python analyze_any_repo.py ../test_project")
        print("   python analyze_any_repo.py /path/to/your/project")
        print("   python analyze_any_repo.py .   # analyze current directory")
        sys.exit(1)

    repo_path = sys.argv[1]
    analyze_repository(repo_path)
