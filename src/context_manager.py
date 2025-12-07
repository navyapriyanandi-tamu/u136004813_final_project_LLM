"""
Simple Context Manager for storing analysis results
Stores AST results and function summaries for module/repo summarization
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path


class ContextManager:
    """
    Manage analysis context for repository summarization
    """

    def __init__(self, repo_name: str):
        """Initialize context for a repository"""
        self.repo_name = repo_name
        self.context = {
            "repo_name": repo_name,
            "ast_results": {},
            "function_summaries": {},
            "module_summaries": {},
            "repo_summary": None,
            "dead_code_analysis": None
        }

    def store_ast_results(self, ast_results: Dict[str, Any]):
        """Store complete AST analysis results"""
        self.context["ast_results"] = ast_results

    def store_function_summaries(self, module_path: str, summaries: Dict[str, Dict[str, str]]):
        """
        Store function summaries for a module
        """
        self.context["function_summaries"][module_path] = summaries

    def get_function_summaries(self, module_path: str) -> Dict[str, Dict[str, str]]:
        """Get function summaries for a specific module"""
        return self.context["function_summaries"].get(module_path, {})

    def get_all_function_summaries(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Get all function summaries across all modules"""
        return self.context["function_summaries"]

    def store_module_summary(self, module_path: str, summary: Dict[str, str]):
        """
        Store module-level summary
        """
        self.context["module_summaries"][module_path] = summary

    def get_module_summary(self, module_path: str) -> Optional[Dict[str, str]]:
        """Get summary for a specific module"""
        return self.context["module_summaries"].get(module_path)

    def get_all_module_summaries(self) -> Dict[str, Dict[str, str]]:
        """Get all module summaries"""
        return self.context["module_summaries"]

    def store_repo_summary(self, summary: Dict[str, str]):
        """
        Store repository-level summary
        """
        self.context["repo_summary"] = summary

    def get_repo_summary(self) -> Optional[Dict[str, str]]:
        """Get repository summary"""
        return self.context["repo_summary"]

    def store_dead_code_analysis(self, analysis: Dict[str, Any]):
        """
        Store dead code analysis results
        """
        self.context["dead_code_analysis"] = analysis

    def get_dead_code_analysis(self) -> Optional[Dict[str, Any]]:
        """Get dead code analysis"""
        return self.context["dead_code_analysis"]

    def get_module_data(self, module_path: str) -> Dict[str, Any]:
        """
        Get complete data for a module (AST + summaries)
        """
        ast_results = self.context["ast_results"]

        return {
            "ast": ast_results.get(module_path, {}),
            "function_summaries": self.get_function_summaries(module_path),
            "module_summary": self.get_module_summary(module_path)
        }

    def save_to_file(self, filepath: str):
        """Save context to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.context, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load context from JSON file"""
        with open(filepath, 'r') as f:
            self.context = json.load(f)

    def get_summary_stats(self) -> Dict[str, int]:
        """Get statistics about stored summaries"""
        total_functions = sum(
            len(summaries)
            for summaries in self.context["function_summaries"].values()
        )

        return {
            "total_modules": len(self.context["ast_results"]) - 1,  
            "modules_with_function_summaries": len(self.context["function_summaries"]),
            "total_function_summaries": total_functions,
            "modules_with_summaries": len(self.context["module_summaries"]),
            "has_repo_summary": self.context["repo_summary"] is not None
        }

    def __repr__(self):
        stats = self.get_summary_stats()
        return f"ContextManager(repo={self.repo_name}, modules={stats['total_modules']}, functions={stats['total_function_summaries']})"
