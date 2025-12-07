"""
Clean AST Parser for Python Repositories
Analyzes code structure, builds dependency graphs, detects dead code
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict
import config


class CodeAnalyzer:
    """
    Python code analyzer using AST
    Analyzes repositories, builds dependency graphs, finds dead code
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.modules: Dict[str, Dict] = {}
        self.all_defined_functions: Set[str] = set()
        self.all_called_functions: Set[str] = set()
        self.all_defined_classes: Set[str] = set()
        self.all_instantiated_classes: Set[str] = set()
        self.exported_symbols: Set[str] = set()  # Tracks public API exports

    def analyze_repository(self) -> Dict[str, Dict]:
        """Analyze entire repository"""
        print(f" Analyzing repository: {self.repo_path}")

        # Find all Python files
        python_files = self._find_python_files()
        print(f"   Found {len(python_files)} Python files")

        # Analyze each file
        for py_file in python_files:
            relative_path = str(py_file.relative_to(self.repo_path))
            print(f"   Analyzing: {relative_path}")

            try:
                analysis = self.analyze_file(py_file)
                if analysis:
                    self.modules[relative_path] = analysis

                    # Track globally defined items
                    for func in analysis.get('functions', []):
                        self.all_defined_functions.add(func['name'])
                    for cls in analysis.get('classes', []):
                        self.all_defined_classes.add(cls['name'])

            except Exception as e:
                print(f"   ⚠️  Error analyzing {relative_path}: {e}")
                self.modules[relative_path] = {'error': str(e)}

        # Build dependency graph and detect dead code
        print(f"\n Building dependency graph")
        self._build_dependency_graph()

        print(f"\n Detecting exports (public API)")
        self._detect_exports()

        print(f"\n Detecting dead code")
        dead_code = self._detect_dead_code()

        # Add summary
        self.modules['__analysis_summary__'] = {
            'total_modules': len(self.modules) - 1,
            'total_functions': len(self.all_defined_functions),
            'total_classes': len(self.all_defined_classes),
            'dead_code_candidates': dead_code
        }

        print(f"\n Analysis complete!")
        print(f"   Modules: {len(self.modules) - 1}")
        print(f"   Functions: {len(self.all_defined_functions)}")
        print(f"   Classes: {len(self.all_defined_classes)}")

        return self.modules

    def analyze_file(self, filepath: Path) -> Optional[Dict]:
        """Analyze a single Python file using AST"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code, filename=str(filepath))

            # Extract all components (pass source code for function code extraction)
            module_docstring = ast.get_docstring(tree)
            functions = self._extract_functions(tree, code)
            classes = self._extract_classes(tree)
            imports = self._extract_imports(tree)
            variables = self._extract_module_variables(tree)
            is_entry_point = self._is_entry_point(tree)
            main_block_calls = self._extract_main_block_calls(tree) if is_entry_point else []
            module_level_calls = self._extract_module_level_calls(tree)  
            dependency_classification = self._classify_dependencies(imports)

            return {
                'filepath': str(filepath),
                'module_docstring': module_docstring,
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'variables': variables,
                'is_entry_point': is_entry_point,
                'main_block_calls': main_block_calls,
                'module_level_calls': module_level_calls,  
                'dependency_classification': dependency_classification,
                'line_count': len(code.split('\n'))
            }

        except (SyntaxError, Exception):
            return None

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in repository"""
        python_files = []
        skip_patterns = set(config.SKIP_PATTERNS)

        for py_file in self.repo_path.rglob("*.py"):
            if any(skip in py_file.parts for skip in skip_patterns):
                continue
            python_files.append(py_file)

        return sorted(python_files)

    def _extract_functions(self, tree, source_code: str) -> List[Dict]:
        """Extract all function definitions with type hints, decorators, and full code"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract arguments with type hints
                args_with_types = []
                for arg in node.args.args:
                    arg_info = {'name': arg.arg}
                    if arg.annotation:
                        arg_info['type'] = self._get_annotation_string(arg.annotation)
                    args_with_types.append(arg_info)

                # Extract return type
                return_type = None
                if node.returns:
                    return_type = self._get_annotation_string(node.returns)

                # Extract decorators with parameters
                decorators = []
                for dec in node.decorator_list:
                    decorators.append(self._extract_decorator_info(dec))

                # Extract function calls and references
                func_calls = []
                for child in ast.walk(node):
                    # Track direct function calls
                    if isinstance(child, ast.Call):
                        call_name = self._get_call_name(child)
                        if call_name:
                            func_calls.append(call_name)

                        # Track function names passed as arguments (callbacks/hooks)
                        for arg in child.args:
                            if isinstance(arg, ast.Name):
                                func_calls.append(arg.id)
                            elif isinstance(arg, ast.Attribute):
                                func_name = self._get_name(arg)
                                if func_name:
                                    func_calls.append(func_name)

                        # Track keyword arguments
                        for keyword in child.keywords:
                            if isinstance(keyword.value, ast.Name):
                                func_calls.append(keyword.value.id)
                            elif isinstance(keyword.value, ast.Attribute):
                                func_name = self._get_name(keyword.value)
                                if func_name:
                                    func_calls.append(func_name)

                    # Track function assignments
                    elif isinstance(child, ast.Assign):
                        if isinstance(child.value, ast.Name):
                            func_calls.append(child.value.id)
                        elif isinstance(child.value, ast.Attribute):
                            func_name = self._get_name(child.value)
                            if func_name:
                                func_calls.append(func_name)

                # Extract full function code
                func_code = ast.get_source_segment(source_code, node)
                if not func_code:
                    func_code = ""

                # Calculate line count
                func_line_count = len(func_code.split('\n')) if func_code else 0

                # Generate summaries for all functions (needed for module/repo summarization)
                # Skip only trivial functions (< 3 lines like simple getters/setters)
                needs_llm_summary = func_line_count >= 3

                functions.append({
                    'name': node.name,
                    'args': args_with_types,
                    'return_type': return_type,
                    'lineno': node.lineno,
                    'decorators': decorators,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'calls': list(set(func_calls)),
                    'docstring': ast.get_docstring(node),
                    'code': func_code,
                    'line_count': func_line_count,
                    'needs_llm_summary': needs_llm_summary
                })

        return functions

    def _extract_classes(self, tree) -> List[Dict]:
        """Extract all class definitions"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(child.name)

                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'base_classes': [self._get_name(base) for base in node.bases],
                    'lineno': node.lineno,
                    'docstring': ast.get_docstring(node)
                })

        return classes

    def _extract_imports(self, tree) -> List[Dict]:
        """Extract all imports"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'lineno': node.lineno,
                        'type': 'import'
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    imports.append({
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'lineno': node.lineno,
                        'type': 'from_import'
                    })

        return imports

    def _extract_module_variables(self, tree) -> List[Dict]:
        """Extract module-level variables with their assignments"""
        variables = []

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                       
                        assigned_to = None
                        if isinstance(node.value, ast.Call):
                            assigned_to = self._get_call_name(node.value)
                        elif isinstance(node.value, ast.Name):
                            assigned_to = node.value.id
                        elif isinstance(node.value, ast.Constant):
                            assigned_to = type(node.value.value).__name__

                        variables.append({
                            'name': target.id,
                            'assigned_to': assigned_to,
                            'lineno': node.lineno
                        })

        return variables

    def _is_entry_point(self, tree) -> bool:
        """Check if module is an entry point"""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                        return True
        return False

    def _extract_main_block_calls(self, tree) -> List[str]:
        """Extract function calls from __main__ block"""
        main_block_calls = []

        for node in tree.body:
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                call_name = self._get_call_name(child)
                                if call_name:
                                    main_block_calls.append(call_name)

        return list(set(main_block_calls))

    def _extract_module_level_calls(self, tree) -> Dict:
        """
        Extract ALL function/method calls and class instantiations at module level
        (outside of function/class definitions)
        """
        module_calls = {
            'function_calls': [],
            'class_instantiations': []
        }

        # Track all nodes to skip (inside function/class definitions)
        skip_nodes = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                skip_nodes.add(id(node))

        # Walk module-level nodes only
        for node in tree.body:
            if id(node) in skip_nodes:
                continue  # Skip function/class definitions themselves

            # Walk all children of this module-level statement
            for child in ast.walk(node):
                # Track function calls
                if isinstance(child, ast.Call):
                    call_name = self._get_call_name(child)
                    if call_name:
                        # Check if it's a class instantiation 
                        if call_name and call_name[0].isupper():
                            
                            base_name = call_name.split('.')[0]
                            module_calls['class_instantiations'].append(base_name)

                        module_calls['function_calls'].append(call_name)

                    # track function names passed as arguments (callbacks/hooks)
                    for arg in child.args:
                        if isinstance(arg, ast.Name):
                            module_calls['function_calls'].append(arg.id)
                        elif isinstance(arg, ast.Attribute):
                            func_name = self._get_name(arg)
                            if func_name:
                                module_calls['function_calls'].append(func_name)

                    # Check keyword arguments
                    for keyword in child.keywords:
                        if isinstance(keyword.value, ast.Name):
                            module_calls['function_calls'].append(keyword.value.id)
                        elif isinstance(keyword.value, ast.Attribute):
                            func_name = self._get_name(keyword.value)
                            if func_name:
                                module_calls['function_calls'].append(func_name)

                # Track function assignments
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(child.value, ast.Name):
                            
                            module_calls['function_calls'].append(child.value.id)
                        elif isinstance(child.value, ast.Attribute):
                           
                            func_name = self._get_name(child.value)
                            if func_name:
                                module_calls['function_calls'].append(func_name)

        
        module_calls['function_calls'] = list(set(module_calls['function_calls']))
        module_calls['class_instantiations'] = list(set(module_calls['class_instantiations']))

        return module_calls

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get function call name, preserving full path for method calls"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_name(node.func)
        return None

    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"

    def _get_annotation_string(self, node) -> str:
        """Convert type annotation to string"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_string(node.value)
            slice_val = self._get_annotation_string(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Tuple):
            elements = [self._get_annotation_string(e) for e in node.elts]
            return ", ".join(elements)
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation_string(node.value)}.{node.attr}"
        return "Any"

    def _extract_decorator_info(self, node) -> Dict:
        """Extract decorator information including parameters"""
        if isinstance(node, ast.Name):
            return {'name': node.id, 'args': [], 'kwargs': {}}

        elif isinstance(node, ast.Call):
            decorator_name = self._get_name(node.func)

            # Extract positional arguments
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
                elif isinstance(arg, ast.List):
                    list_items = [item.value for item in arg.elts if isinstance(item, ast.Constant)]
                    args.append(list_items)

            # Extract keyword arguments
            kwargs = {}
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    kwargs[keyword.arg] = keyword.value.value
                elif isinstance(keyword.value, ast.List):
                    list_items = [item.value for item in keyword.value.elts if isinstance(item, ast.Constant)]
                    kwargs[keyword.arg] = list_items

            return {'name': decorator_name, 'args': args, 'kwargs': kwargs}

        elif isinstance(node, ast.Attribute):
            return {'name': self._get_name(node), 'args': [], 'kwargs': {}}

        return {'name': 'unknown', 'args': [], 'kwargs': {}}

    def _classify_dependencies(self, imports: List[Dict]) -> Dict:
        """Classify imports into stdlib vs third-party"""
        STDLIB_MODULES = {
            'os', 'sys', 'pathlib', 'datetime', 'time', 'json', 'csv', 'argparse',
            'logging', 'collections', 're', 'subprocess', 'shutil', 'glob',
            'typing', 'dataclasses', 'functools', 'itertools', 'math', 'random',
            'io', 'tempfile', 'unittest', 'asyncio', 'threading', 'multiprocessing',
            'sqlite3', 'pickle', 'copy', 'enum', 'abc', 'contextlib', 'warnings'
        }

        THIRD_PARTY_CATEGORIES = {
            'web': {'flask', 'django', 'fastapi', 'requests', 'aiohttp'},
            'ai': {'anthropic', 'openai', 'langchain', 'transformers', 'torch', 'tensorflow'},
            'data': {'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn'},
            'database': {'sqlalchemy', 'pymongo', 'redis', 'psycopg2'},
            'testing': {'pytest', 'mock', 'coverage'},
            'config': {'pydantic', 'dotenv', 'yaml'}
        }

        stdlib = []
        third_party = {}
        tech_stack = set()

        for imp in imports:
            module_name = imp.get('module', '').split('.')[0]
            if not module_name:
                module_name = imp.get('name', '').split('.')[0]

            if module_name in STDLIB_MODULES:
                stdlib.append(module_name)
            else:
                categorized = False
                for category, modules in THIRD_PARTY_CATEGORIES.items():
                    if module_name in modules:
                        if category not in third_party:
                            third_party[category] = []
                        if module_name not in third_party[category]:
                            third_party[category].append(module_name)
                        tech_stack.add(category.capitalize())
                        categorized = True
                        break

                if not categorized and module_name:
                    if 'other' not in third_party:
                        third_party['other'] = []
                    if module_name not in third_party['other']:
                        third_party['other'].append(module_name)

        return {
            'stdlib': list(set(stdlib)),
            'third_party': third_party,
            'tech_stack': list(tech_stack)
        }

    def _build_dependency_graph(self):
        """Build dependency graph showing what each module calls"""
        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            # Track function calls from within functions
            for func in data.get('functions', []):
                func_calls = func.get('calls', [])
                self.all_called_functions.update(func_calls)

                # track bare function names from qualified calls
                
                for call in func_calls:
                    if '.' in call:
                        bare_name = call.split('.')[-1]
                        self.all_called_functions.add(bare_name)

                # Track class instantiations
                for call in func_calls:
                    if call in self.all_defined_classes:
                        self.all_instantiated_classes.add(call)

            # Track module-level calls (code outside functions/classes)
            module_level_calls = data.get('module_level_calls', {})
            if module_level_calls:
                # Add function calls from module level
                func_calls = module_level_calls.get('function_calls', [])
                self.all_called_functions.update(func_calls)

                # Add bare names from qualified calls
                for call in func_calls:
                    if '.' in call:
                        bare_name = call.split('.')[-1]
                        self.all_called_functions.add(bare_name)

                # Add class instantiations from module level
                class_insts = module_level_calls.get('class_instantiations', [])
                self.all_instantiated_classes.update(class_insts)

                # check if instantiation names match defined classes
                for call in func_calls:
                    if call in self.all_defined_classes:
                        self.all_instantiated_classes.add(call)

    def _detect_exports(self):
        """
        Detect symbols exported via __init__.py or __all__ declarations

        Libraries export functions/classes for external use via:
        1. from .module import function (in __init__.py)
        2. __all__ = ['function', 'Class'] declarations
        3. Re-exports in package __init__ files
        4. Config files consumed by external tools

        These should NOT be flagged as dead code even if unused internally.
        """
        # Track __all__ declarations and config files
        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            # Config files: variables consumed by external tools
            config_files = ['conf.py', 'setup.py', 'conftest.py', '__version__.py']
            is_config_file = any(module_path.endswith(cf) for cf in config_files)

            if is_config_file:
                # All variables in config files are considered used by external tools
                for var in data.get('variables', []):
                    self.exported_symbols.add(var['name'])

                # All imports in config files may be needed
                for imp in data.get('imports', []):
                    if imp.get('type') == 'from_import':
                        imported_name = imp.get('name', '')
                        if imported_name:
                            self.exported_symbols.add(imported_name)

            # Check for __all__ variable
            for var in data.get('variables', []):
                if var['name'] == '__all__':
                    # This marks module as having explicit exports
                    # Need to parse the actual list values
                    pass

            # Check __init__.py files for imports (public API exports)
            if module_path.endswith('__init__.py'):
                # All imports in __init__.py are considered exports
                for imp in data.get('imports', []):
                    if imp.get('type') == 'from_import':
                        # from .module import function -> 'function' is exported
                        imported_name = imp.get('name', '')
                        if imported_name:
                            self.exported_symbols.add(imported_name)
                    elif imp.get('type') == 'import':
                        # import module -> 'module' is exported
                        module_name = imp.get('module', '')
                        if module_name:
                            self.exported_symbols.add(module_name)

                # Variables imported in __init__.py are also exported
              
                for var in data.get('variables', []):
                    var_name = var['name']
                    # Common metadata variables
                    if var_name.startswith('__') and var_name.endswith('__'):
                        self.exported_symbols.add(var_name)

    def _detect_dead_code(self) -> Dict:
        """
        Detect potential dead code across multiple categories

        Returns dict with:
        - unreferenced_functions: Functions never called
        - unused_classes: Classes never instantiated
        - unused_imports: Imports never referenced (no LLM needed)
        - unused_global_variables: Module-level variables never read (LLM recommended)
        - unreachable_code: Code after return/break/etc (LLM recommended)
        - suspicious_patterns: Code smells needing review (LLM recommended)
        """
        dead_code = {
            'unreferenced_functions': [],
            'unused_classes': [],
            'unused_imports': [],              
            'unused_global_variables': [],     
            'unreachable_code': [],            
            'suspicious_patterns': []          
        }

        # Find unreferenced functions
        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            for func in data.get('functions', []):
                func_name = func['name']

                # Skip special methods and entry points
                if func_name.startswith('__') or func_name in ['main', 'setUp', 'tearDown']:
                    continue

                # Skip if in entry point module
                if data.get('is_entry_point') and func_name == 'main':
                    continue

                # Skip functions with decorators (often called dynamically)
                
                decorators = func.get('decorators', [])
                if decorators:
                    # Check if any decorator suggests dynamic calling or special access
                    dynamic_decorators = ['tool', 'route', 'list_tools', 'call_tool', 'endpoint',
                                         'property', 'cached_property', 'classmethod', 'staticmethod']
                    has_dynamic_decorator = any(
                        any(dec_keyword in dec.get('name', '').lower() for dec_keyword in dynamic_decorators)
                        for dec in decorators
                    )
                    if has_dynamic_decorator:
                        continue

                # Skip framework-specific implicit methods
               
                framework_implicit_methods = [
                    'forward',      # PyTorch nn.Module.__call__() -> forward()
                    'call',         # TensorFlow/Keras Model.__call__() -> call()
                    'predict',      # Scikit-learn estimators
                    'fit',          # Scikit-learn estimators
                    'transform',    # Scikit-learn transformers
                    'render',       # Django views
                    'dispatch',     # Flask/Django dispatch
                ]

                if func_name in framework_implicit_methods:
                    # Check if this is inside a class (method)
                    
                    is_in_class = False
                    for cls in data.get('classes', []):
                        methods = cls.get('methods', [])
                        if isinstance(methods, list) and func_name in methods:
                            is_in_class = True
                            break
                    if is_in_class:
                        continue  

                # Skip interface/protocol methods in classes with known base classes
                
                interface_methods = [
                    'get_type', 'get_host', 'get_full_url', 'get_header', 'has_header',
                    'add_header', 'add_unredirected_header', 'is_unverifiable',
                    'origin_req_host', 'unverifiable', 'host',  # urllib2.Request interface
                    'keys', 'values', 'items', 'get',  # dict-like interface
                    '__iter__', '__next__', '__len__', '__getitem__', '__setitem__',  # Protocol methods
                ]

                if func_name in interface_methods:
                    # Check if this function is in a class that inherits from stdlib/external
                    is_interface_method = False
                    for cls in data.get('classes', []):
                        methods = cls.get('methods', [])
                        if isinstance(methods, list) and func_name in methods:
                            # Check if class has base classes (likely interface implementation)
                            base_classes = cls.get('base_classes', [])
                            if base_classes:  
                                is_interface_method = True
                                break
                    if is_interface_method:
                        continue  

                # Skip if exported (public API)
                if func_name in self.exported_symbols:
                    continue

                # Check if never called
                if func_name not in self.all_called_functions:
                    dead_code['unreferenced_functions'].append({
                        'name': func_name,
                        'module': module_path,
                        'lineno': func['lineno']
                    })

        # Find unused classes
        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            for cls in data.get('classes', []):
                cls_name = cls['name']

                # Skip if exported (public API)
                if cls_name in self.exported_symbols:
                    continue

                if cls_name not in self.all_instantiated_classes:
                    dead_code['unused_classes'].append({
                        'name': cls_name,
                        'module': module_path,
                        'lineno': cls['lineno']
                    })

        # Detect unused imports
        dead_code['unused_imports'] = self._detect_unused_imports()

        # Detect unused global variables
        dead_code['unused_global_variables'] = self._detect_unused_global_variables()

        # Detect unreachable code
        dead_code['unreachable_code'] = self._detect_unreachable_code()

        # Detect suspicious patterns
        dead_code['suspicious_patterns'] = self._detect_suspicious_patterns()

        return dead_code

    def _detect_unused_imports(self) -> List[Dict]:
        """
        Detect imports that are never used in the code

        Strategy:
        1. For each import, track what symbols it provides
        2. Search all code for usage of those symbols
        3. Flag imports with no usage

        Returns: List of {module, import_statement, lineno, symbols, needs_llm}
        """
        unused = []

        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            imports = data.get('imports', [])
            if not imports:
                continue

            # Collect all symbols used in this module
            used_symbols = set()

            # From function calls (all levels)
            for func in data.get('functions', []):
                used_symbols.update(func.get('calls', []))

            # From type hints
            for func in data.get('functions', []):
                for arg in func.get('args', []):
                    if arg.get('type'):
                        used_symbols.add(arg['type'])
                if func.get('return_type'):
                    used_symbols.add(func['return_type'])

            # From module-level calls
            module_level = data.get('module_level_calls', {})
            if module_level:
                used_symbols.update(module_level.get('function_calls', []))

            # From class base classes
            for cls in data.get('classes', []):
                used_symbols.update(cls.get('base_classes', []))

            # Check each import
            for imp in imports:
                module_name = imp.get('module', '')
                import_type = imp.get('type', 'import')
                alias = imp.get('alias')

                if not module_name:
                    continue

                # Check if import is used
                is_used = False

                # Case 1: "import os" or "import torch.nn.functional as F"
                if import_type == 'import':
                    # If there's an alias, check for alias usage; otherwise check module name
                    name_to_check = alias if alias else module_name

                    # Check if the name (alias or module) appears in any symbol
                    for sym in used_symbols:
                        if name_to_check in sym:
                            is_used = True
                            break
                    symbols_to_report = [name_to_check]

                # Case 2: "from x import Y" or "from x import Y as Z"
                elif import_type == 'from_import':
                    imported_name = imp.get('name', '')
                    if imported_name:
                        # If there's an alias, check for alias usage; otherwise check imported name
                        name_to_check = alias if alias else imported_name

                        # Check if the name (alias or imported) is used
                        if name_to_check in used_symbols or any(name_to_check in sym for sym in used_symbols):
                            is_used = True
                        symbols_to_report = [name_to_check]
                    else:
                        symbols_to_report = [module_name]
                else:
                    symbols_to_report = [module_name]

                # Skip if exported (imported in __init__.py or config file)
                if any(sym in self.exported_symbols for sym in symbols_to_report):
                    continue

                if not is_used:
                    unused.append({
                        'module': module_path,
                        'import_statement': imp.get('statement', f"import {module_name}"),
                        'lineno': imp.get('lineno', 0),
                        'imported_module': module_name,
                        'symbols': symbols_to_report,
                        'needs_llm': False  # Unused imports don't need LLM explanation
                    })

        return unused

    def _detect_unused_global_variables(self) -> List[Dict]:
        """
        Detect module-level variables that are never read

        Only flags GLOBAL variables (module-level), not local variables.
        These are valuable for LLM analysis because they might be:
        - Configuration constants used externally
        - Part of public API
        - Legacy code to be cleaned up

        Returns: List of {module, name, lineno, assigned_to, needs_llm}
        """
        unused = []

        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            variables = data.get('variables', [])
            if not variables:
                continue

            # Get the actual file path and parse AST to find all Name nodes
            filepath = data.get('filepath', '')
            if not filepath or not Path(filepath).exists():
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                tree = ast.parse(source_code, filename=filepath)
            except:
                continue

            # Collect all variable READS (Load context) across the module
            reads = set()

            # Walk the entire AST looking for Name nodes with Load context
            for node in ast.walk(tree):
                # Direct variable reads: x, dataset, batch_size, etc.
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    reads.add(node.id)

                # Attribute access: dataset.num_features -> reads 'dataset'
                elif isinstance(node, ast.Attribute):
                    # Get the base object being accessed
                    if isinstance(node.value, ast.Name):
                        reads.add(node.value.id)

            # Flag variables that are only written, never read
            for var in variables:
                var_name = var['name']

                # Skip special variables
                if var_name.startswith('__') and var_name.endswith('__'):
                    continue  

                # Skip if exported (public API or config file)
                if var_name in self.exported_symbols:
                    continue

                if var_name not in reads:
                    unused.append({
                        'module': module_path,
                        'name': var_name,
                        'lineno': var.get('lineno', 0),
                        'assigned_to': var.get('assigned_to', 'unknown'),
                        'needs_llm': True  # LLM can explain if it's config, API, or dead code
                    })

        return unused

    def _detect_unreachable_code(self) -> List[Dict]:
        """
        Detect code that can never be executed

        Patterns detected:
        1. Code after return/raise/break/continue
        2. if False: blocks
        3. while False: loops
        4. Code after infinite loops

        These are HIGH VALUE for LLM because they need explanation:
        - Why is it there? (debugging, disabled feature, mistake)
        - Safe to delete or document?

        Returns: List of {module, function, lineno, reason, code_preview, needs_llm}
        """
        unreachable = []

        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            # We need to re-parse the AST to detect unreachable code
            
            filepath = data.get('filepath', '')
            if not filepath or not Path(filepath).exists():
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()
                tree = ast.parse(code, filename=filepath)

                # Analyze each function for unreachable code
                for func_def in ast.walk(tree):
                    if not isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue

                    func_name = func_def.name
                    unreachable_in_func = self._find_unreachable_in_function(func_def, code)

                    for item in unreachable_in_func:
                        item['module'] = module_path
                        item['function'] = func_name
                        item['needs_llm'] = True  # Definitely needs explanation
                        unreachable.append(item)

                # Check for module-level unreachable code
                for node in tree.body:
                    if isinstance(node, ast.If):
                        # Check for "if False:" pattern
                        if isinstance(node.test, ast.Constant) and node.test.value is False:
                            unreachable.append({
                                'module': module_path,
                                'function': '<module-level>',
                                'lineno': node.lineno,
                                'reason': 'Code in "if False:" block',
                                'code_preview': ast.unparse(node) if hasattr(ast, 'unparse') else 'if False: ...',
                                'needs_llm': True
                            })

            except Exception:
               
                continue

        return unreachable

    def _find_unreachable_in_function(self, func_node, source_code: str) -> List[Dict]:
        """Helper to find unreachable code within a function"""
        unreachable = []

        # Walk through function body
        statements = func_node.body
        found_terminal = False

        for i, stmt in enumerate(statements):
            # Check if previous statement was terminal (return, raise, etc.)
            if found_terminal:
                unreachable.append({
                    'lineno': stmt.lineno,
                    'reason': 'Code after return/raise/break/continue',
                    'code_preview': source_code.split('\n')[stmt.lineno - 1].strip()[:80] if stmt.lineno <= len(source_code.split('\n')) else '...'
                })

            # Check if this statement is terminal
            if isinstance(stmt, (ast.Return, ast.Raise)):
                found_terminal = True
            elif isinstance(stmt, (ast.Break, ast.Continue)):
                found_terminal = True

        return unreachable

    def _detect_suspicious_patterns(self) -> List[Dict]:
        """
        Detect code patterns that might indicate problems or need review

        Patterns detected:
        1. Functions called only once (inline candidates)
        2. Empty/placeholder functions (pass, return None only)
        3. Bare except: clauses (catch-all exception handlers)
        4. Similar function names (possible duplicates)
        5. Very long functions (>100 lines - code smell)
        6. Functions with many parameters (>6 args - code smell)

        These are VERY HIGH VALUE for LLM because architectural insights needed.

        Returns: List of {module, name, lineno, pattern_type, details, needs_llm}
        """
        suspicious = []

        # Pattern 1: Functions called only once (from a single location)
        # Need to actually walk the AST to count distinct call locations
        function_call_sites = {}  # func_name -> count of distinct call sites

        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            # Count calls within each function
            for func in data.get('functions', []):
                for call in func.get('calls', []):
                    # Extract base name 
                    base_name = call.split('.')[-1]
                    function_call_sites[base_name] = function_call_sites.get(base_name, 0) + 1
                    if call != base_name:
                        function_call_sites[call] = function_call_sites.get(call, 0) + 1

            # Count calls at module level
            module_level = data.get('module_level_calls', {})
            if module_level:
                for call in module_level.get('function_calls', []):
                    base_name = call.split('.')[-1]
                    function_call_sites[base_name] = function_call_sites.get(base_name, 0) + 1
                    if call != base_name:
                        function_call_sites[call] = function_call_sites.get(call, 0) + 1

        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue

            functions = data.get('functions', [])

            for func in functions:
                func_name = func['name']
                lineno = func.get('lineno', 0)

                # Skip special methods and constructors (normal to be called once)
                if func_name.startswith('__') and func_name.endswith('__'):
                    continue  # Skip __init__, __str__, __call__, etc.

                # Pattern 2: Empty/placeholder functions
                code = func.get('code', '')
                lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
                # Check if function body is just 'pass' or 'return None'
                if len(lines) <= 2:  
                    if 'pass' in code or code.strip().endswith('return None'):
                        suspicious.append({
                            'module': module_path,
                            'name': func_name,
                            'lineno': lineno,
                            'pattern_type': 'empty_function',
                            'details': 'Function is empty or only contains pass/return None',
                            'needs_llm': True
                        })

                # Pattern 5: Very long functions
                line_count = func.get('line_count', 0)
                if line_count > 100:
                    suspicious.append({
                        'module': module_path,
                        'name': func_name,
                        'lineno': lineno,
                        'pattern_type': 'long_function',
                        'details': f'Function is very long ({line_count} lines) - consider refactoring',
                        'needs_llm': True
                    })

                # Pattern 6: Too many parameters
                args_count = len(func.get('args', []))
                if args_count > 6:
                    suspicious.append({
                        'module': module_path,
                        'name': func_name,
                        'lineno': lineno,
                        'pattern_type': 'too_many_parameters',
                        'details': f'Function has {args_count} parameters - consider parameter object',
                        'needs_llm': True
                    })

        # Pattern 4: Similar function names (possible duplicates)
        all_functions = []
        for module_path, data in self.modules.items():
            if module_path == '__analysis_summary__' or 'error' in data:
                continue
            for func in data.get('functions', []):
                all_functions.append((module_path, func['name'], func.get('lineno', 0)))

        # Find similar names
        for i, (mod1, name1, line1) in enumerate(all_functions):
            for mod2, name2, line2 in all_functions[i+1:]:
                # Check for similar names 
                if self._are_names_similar(name1, name2):
                    suspicious.append({
                        'module': mod1,
                        'name': name1,
                        'lineno': line1,
                        'pattern_type': 'similar_function_names',
                        'details': f'Similar to {name2} in {mod2} - possible duplicate logic',
                        'needs_llm': True
                    })

        return suspicious

    def _are_names_similar(self, name1: str, name2: str) -> bool:
        """Check if two function names are suspiciously similar"""
        # Skip if same name
        if name1 == name2:
            return False

        # Check for common prefix/suffix patterns
        
        if name1.startswith(name2) or name2.startswith(name1):
            return True

        # Check for similar with underscore variations
       
        if name1.lower().replace('_', '') == name2.lower().replace('_', ''):
            return True

        return False

    def get_summary(self) -> Dict:
        """Get analysis summary"""
        return self.modules.get('__analysis_summary__', {})

    def get_dead_code_candidates(self) -> Dict:
        """Get dead code candidates"""
        summary = self.get_summary()
        return summary.get('dead_code_candidates', {})


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ast_parser_clean <path_to_repo>")
        sys.exit(1)

    repo_path = sys.argv[1]
    if not Path(repo_path).exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    analyzer = CodeAnalyzer(repo_path)
    results = analyzer.analyze_repository()

    summary = analyzer.get_summary()
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total modules: {summary.get('total_modules', 0)}")
    print(f"Total functions: {summary.get('total_functions', 0)}")
    print(f"Total classes: {summary.get('total_classes', 0)}")

    dead_code = summary.get('dead_code_candidates', {})
    print(f"\nDead code candidates:")
    print(f"  Unreferenced functions: {len(dead_code.get('unreferenced_functions', []))}")
    print(f"  Unused classes: {len(dead_code.get('unused_classes', []))}")
