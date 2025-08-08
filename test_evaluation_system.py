#!/usr/bin/env python3
"""
Test script for the evaluation system.
Tests both configuration loading and evaluation execution with various scenarios.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set test environment variables
os.environ['OPENAI_API_KEY'] = 'test-key-for-validation'
os.environ['ANTHROPIC_API_KEY'] = 'test-key-for-validation'
os.environ['GOOGLE_API_KEY'] = 'test-key-for-validation'
os.environ['GROQ_API_KEY'] = 'test-key-for-validation'

try:
    from src.core.configurable_agent import ConfigurableAgent
    from src.core.config_loader import ConfigLoader
    from src.evaluation.evaluation_manager import EvaluationManager
    from src.evaluation.evaluators.built_in import (
        CorrectnessEvaluator, HelpfulnessEvaluator, 
        ResponseTimeEvaluator, ToolUsageEvaluator
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


class EvaluationSystemTester:
    """Test suite for the evaluation system."""
    
    def __init__(self):
        self.test_results = []
        self.configs_tested = 0
        self.evaluations_tested = 0
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log a test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": time.time()
        })
    
    def test_config_loading(self) -> bool:
        """Test evaluation configuration loading."""
        print("🔍 Testing evaluation configuration loading...")
        
        configs_to_test = [
            'configs/examples/evaluation_enabled_agent.yml',
            'configs/examples/research_agent.yml',
            'configs/examples/coding_assistant.yml'
        ]
        
        all_passed = True
        
        for config_path in configs_to_test:
            try:
                if not os.path.exists(config_path):
                    self.log_test_result(f"Config exists: {config_path}", False, "File not found")
                    all_passed = False
                    continue
                
                config_loader = ConfigLoader()
                config = config_loader.load_config(config_path)
                
                # Test config validation
                validation = config_loader.validate_evaluation_config()
                
                self.log_test_result(
                    f"Config loading: {config_path}",
                    True,
                    f"Evaluators: {validation['total_evaluators']}, Valid: {validation['valid']}"
                )
                
                if validation['warnings']:
                    print(f"   ⚠️ Warnings: {'; '.join(validation['warnings'][:2])}")
                
                self.configs_tested += 1
                
            except Exception as e:
                self.log_test_result(f"Config loading: {config_path}", False, str(e))
                all_passed = False
        
        return all_passed
    
    def test_built_in_evaluators(self) -> bool:
        """Test built-in evaluators with sample data."""
        print("🔍 Testing built-in evaluators...")
        
        # Sample test data
        test_cases = [
            {
                "name": "Research Query",
                "inputs": {"query": "What is the capital of France?", "question": "What is the capital of France?"},
                "outputs": {
                    "response": "The capital of France is Paris. Paris is located in the north-central part of France.",
                    "response_time": 2.5,
                    "tool_results": {"web_search": {"result": "Paris is the capital of France"}},
                    "iteration_count": 1
                },
                "reference": {"answer": "Paris"}
            },
            {
                "name": "Math Query",
                "inputs": {"query": "Calculate 25 * 4", "question": "Calculate 25 * 4"},
                "outputs": {
                    "response": "25 * 4 = 100",
                    "response_time": 1.2,
                    "tool_results": {"calculator": {"result": 100}},
                    "iteration_count": 1
                },
                "reference": {"answer": "100"}
            },
            {
                "name": "Complex Query (No Tools Needed)",
                "inputs": {"query": "Explain photosynthesis", "question": "Explain photosynthesis"},
                "outputs": {
                    "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
                    "response_time": 3.8,
                    "tool_results": {},
                    "iteration_count": 0
                }
            }
        ]
        
        evaluators = {
            "correctness": CorrectnessEvaluator(),
            "helpfulness": HelpfulnessEvaluator(), 
            "response_time": ResponseTimeEvaluator(),
            "tool_usage": ToolUsageEvaluator()
        }
        
        all_passed = True
        
        for test_case in test_cases:
            print(f"  Testing: {test_case['name']}")
            
            for evaluator_name, evaluator in evaluators.items():
                try:
                    start_time = time.time() - test_case["outputs"]["response_time"]
                    end_time = time.time()
                    
                    result = evaluator.evaluate(
                        inputs=test_case["inputs"],
                        outputs=test_case["outputs"],
                        reference_outputs=test_case.get("reference"),
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    # Validate result structure
                    if not isinstance(result, dict) or "score" not in result:
                        self.log_test_result(f"  {evaluator_name} structure", False, "Invalid result structure")
                        all_passed = False
                        continue
                    
                    score = result["score"]
                    if not (0.0 <= score <= 1.0):
                        self.log_test_result(f"  {evaluator_name} score range", False, f"Score {score} not in [0,1]")
                        all_passed = False
                        continue
                    
                    self.log_test_result(
                        f"  {evaluator_name}",
                        True,
                        f"Score: {score:.2f}, Reasoning: {result.get('reasoning', 'N/A')[:50]}..."
                    )
                    
                    self.evaluations_tested += 1
                    
                except Exception as e:
                    self.log_test_result(f"  {evaluator_name}", False, str(e))
                    all_passed = False
        
        return all_passed
    
    def test_evaluation_manager(self) -> bool:
        """Test the evaluation manager with mock configurations."""
        print("🔍 Testing evaluation manager...")
        
        try:
            # Create a minimal evaluation configuration
            from src.core.config_loader import EvaluationConfig, EvaluatorConfig, LangSmithConfig
            
            eval_config = EvaluationConfig(
                enabled=True,
                langsmith=LangSmithConfig(enabled=False),  # Disable LangSmith for testing
                evaluators=[
                    EvaluatorConfig(name="correctness", type="heuristic", enabled=True),
                    EvaluatorConfig(name="helpfulness", type="heuristic", enabled=True),
                    EvaluatorConfig(name="response_time", type="heuristic", enabled=True),
                ],
                auto_evaluate=True,
                evaluation_frequency="per_run"
            )
            
            # Initialize evaluation manager
            eval_manager = EvaluationManager(eval_config)
            
            # Test single evaluation
            input_data = {"query": "Test query", "input": "Test input"}
            output_data = {
                "response": "This is a test response that provides helpful information.",
                "response_time": 2.1,
                "tool_results": {},
                "iteration_count": 1,
                "metadata": {"agent_name": "test_agent"}
            }
            
            results = eval_manager.evaluate_single(
                input_data=input_data,
                output_data=output_data,
                start_time=time.time() - 2.1,
                end_time=time.time()
            )
            
            # Validate results
            if not isinstance(results, dict) or len(results) == 0:
                self.log_test_result("Evaluation manager", False, "No results returned")
                return False
            
            # Check that all expected evaluators ran
            expected_evaluators = ["correctness", "helpfulness", "response_time"]
            for evaluator in expected_evaluators:
                if evaluator not in results:
                    self.log_test_result("Evaluation manager", False, f"Missing evaluator: {evaluator}")
                    return False
                
                if not isinstance(results[evaluator], dict) or "score" not in results[evaluator]:
                    self.log_test_result("Evaluation manager", False, f"Invalid result for {evaluator}")
                    return False
            
            # Test metrics collection
            metrics_summary = eval_manager.get_metrics_summary()
            if not isinstance(metrics_summary, dict):
                self.log_test_result("Metrics collection", False, "Invalid metrics summary")
                return False
            
            self.log_test_result(
                "Evaluation manager",
                True,
                f"Ran {len(results)} evaluators, collected metrics"
            )
            
            return True
            
        except Exception as e:
            self.log_test_result("Evaluation manager", False, str(e))
            return False
    
    def test_agent_integration(self) -> bool:
        """Test evaluation integration with a real agent configuration."""
        print("🔍 Testing agent integration...")
        
        try:
            # Use evaluation_enabled_agent.yml if available
            config_path = 'configs/examples/evaluation_enabled_agent.yml'
            if not os.path.exists(config_path):
                self.log_test_result("Agent integration", False, f"Config not found: {config_path}")
                return False
            
            # Temporarily disable LangSmith to avoid API key issues
            with open(config_path, 'r') as f:
                import yaml
                config_data = yaml.safe_load(f)
            
            # Disable LangSmith for testing
            if 'evaluation' in config_data and 'langsmith' in config_data['evaluation']:
                config_data['evaluation']['langsmith']['enabled'] = False
            
            # Set auto_evaluate to True for testing
            config_data['evaluation']['auto_evaluate'] = True
            config_data['evaluation']['evaluation_frequency'] = 'per_run'
            
            # Write temporary config
            temp_config_path = 'temp_test_config.yml'
            with open(temp_config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            try:
                # Initialize agent
                agent = ConfigurableAgent(temp_config_path)
                
                # Check evaluation manager was initialized
                if not agent.evaluation_manager:
                    self.log_test_result("Agent evaluation init", False, "Evaluation manager not initialized")
                    return False
                
                # Test manual evaluation
                test_input = "What is artificial intelligence?"
                evaluation_result = agent.evaluate_single(test_input)
                
                if not isinstance(evaluation_result, dict) or len(evaluation_result) == 0:
                    self.log_test_result("Agent evaluation", False, "No evaluation results")
                    return False
                
                self.log_test_result(
                    "Agent integration",
                    True,
                    f"Evaluated with {len(evaluation_result)} evaluators"
                )
                
                return True
                
            finally:
                # Clean up temporary config
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
                    
        except Exception as e:
            self.log_test_result("Agent integration", False, str(e))
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all evaluation system tests."""
        print("🚀 Starting Evaluation System Tests")
        print("=" * 50)
        
        test_methods = [
            ("Configuration Loading", self.test_config_loading),
            ("Built-in Evaluators", self.test_built_in_evaluators),
            ("Evaluation Manager", self.test_evaluation_manager),
            ("Agent Integration", self.test_agent_integration)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            print(f"\n📋 {test_name}")
            print("-" * 30)
            
            try:
                success = test_method()
                if success:
                    passed_tests += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} FAILED: {str(e)}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 EVALUATION SYSTEM TEST SUMMARY")
        print("=" * 50)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Configurations Tested: {self.configs_tested}")
        print(f"Evaluations Performed: {self.evaluations_tested}")
        
        # Show failed tests
        failed_tests = [r for r in self.test_results if not r["success"]]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"  - {test['test_name']}: {test['details']}")
        
        success_rate = passed_tests / total_tests
        overall_status = "🎉 ALL TESTS PASSED!" if success_rate == 1.0 else f"⚠️  {success_rate:.1%} Tests Passed"
        print(f"\n{overall_status}")
        
        return {
            "success_rate": success_rate,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "configs_tested": self.configs_tested,
            "evaluations_performed": self.evaluations_tested,
            "failed_tests": failed_tests,
            "details": self.test_results
        }


if __name__ == "__main__":
    tester = EvaluationSystemTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["success_rate"] == 1.0 else 1)