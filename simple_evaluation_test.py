#!/usr/bin/env python3
"""
Simple test for evaluation system without external dependencies.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_evaluator_imports():
    """Test that we can import all evaluator components."""
    print("🔍 Testing evaluator imports...")
    
    try:
        from src.evaluation.evaluators.built_in import (
            CorrectnessEvaluator, HelpfulnessEvaluator, 
            ResponseTimeEvaluator, ToolUsageEvaluator
        )
        print("✅ Built-in evaluators import successful")
        return True
    except ImportError as e:
        print(f"❌ Failed to import built-in evaluators: {e}")
        return False

def test_individual_evaluators():
    """Test individual evaluators with mock data."""
    print("🔍 Testing individual evaluators...")
    
    try:
        from src.evaluation.evaluators.built_in import (
            CorrectnessEvaluator, HelpfulnessEvaluator, 
            ResponseTimeEvaluator, ToolUsageEvaluator
        )
        
        # Test data
        inputs = {"query": "What is the capital of France?", "question": "What is the capital of France?"}
        outputs = {
            "response": "The capital of France is Paris.",
            "output": "The capital of France is Paris.",
            "response_time": 2.5,
            "tool_results": {"web_search": {"result": "Paris"}},
            "iteration_count": 1,
            "metadata": {"tools_available": 3}
        }
        reference = {"answer": "Paris"}
        
        evaluators = [
            ("Correctness", CorrectnessEvaluator()),
            ("Helpfulness", HelpfulnessEvaluator()),
            ("Response Time", ResponseTimeEvaluator()),
            ("Tool Usage", ToolUsageEvaluator())
        ]
        
        all_passed = True
        for name, evaluator in evaluators:
            try:
                result = evaluator.evaluate(
                    inputs=inputs,
                    outputs=outputs,
                    reference_outputs=reference,
                    start_time=time.time() - 2.5,
                    end_time=time.time()
                )
                
                # Check result structure
                if isinstance(result, dict) and "score" in result and "reasoning" in result:
                    score = result["score"]
                    if 0.0 <= score <= 1.0:
                        print(f"✅ {name}: Score {score:.2f} - {result['reasoning'][:50]}...")
                    else:
                        print(f"❌ {name}: Invalid score {score}")
                        all_passed = False
                else:
                    print(f"❌ {name}: Invalid result structure")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ {name}: Error - {str(e)}")
                all_passed = False
        
        return all_passed
    
    except Exception as e:
        print(f"❌ Evaluator testing failed: {e}")
        return False

def test_evaluation_manager_creation():
    """Test evaluation manager creation without dependencies."""
    print("🔍 Testing evaluation manager creation...")
    
    try:
        from src.evaluation.evaluation_manager import EvaluationManager
        from src.core.config_loader import EvaluationConfig, LangSmithConfig, EvaluatorConfig
        
        # Create minimal config
        config = EvaluationConfig(
            enabled=True,
            langsmith=LangSmithConfig(enabled=False),
            evaluators=[
                EvaluatorConfig(name="correctness", type="heuristic", enabled=True)
            ],
            auto_evaluate=False
        )
        
        # Create evaluation manager
        eval_manager = EvaluationManager(config)
        
        if eval_manager and len(eval_manager.evaluators) > 0:
            print(f"✅ Evaluation manager created with {len(eval_manager.evaluators)} evaluators")
            return True
        else:
            print("❌ Evaluation manager creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation manager test failed: {e}")
        return False

def test_config_validation():
    """Test configuration validation."""
    print("🔍 Testing configuration validation...")
    
    try:
        from src.core.config_loader import ConfigLoader, EvaluationConfig, EvaluatorConfig
        
        # Test with empty config
        config_loader = ConfigLoader()
        config_loader._config = type('MockConfig', (), {
            'evaluation': EvaluationConfig(
                enabled=True,
                evaluators=[
                    EvaluatorConfig(name="test", type="heuristic", enabled=True)
                ]
            )
        })()
        
        validation = config_loader.validate_evaluation_config()
        
        if validation and isinstance(validation, dict) and "valid" in validation:
            print(f"✅ Config validation works: {validation['total_evaluators']} evaluators")
            return True
        else:
            print("❌ Config validation failed")
            return False
    
    except Exception as e:
        print(f"❌ Config validation test failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("🚀 Simple Evaluation System Tests")
    print("=" * 40)
    
    tests = [
        ("Evaluator Imports", test_evaluator_imports),
        ("Individual Evaluators", test_individual_evaluators),
        ("Evaluation Manager", test_evaluation_manager_creation),
        ("Config Validation", test_config_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 25)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {str(e)}")
    
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {passed/total:.1%} Tests Passed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)