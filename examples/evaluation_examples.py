#!/usr/bin/env python3
"""
Comprehensive examples demonstrating LangSmith evaluation integration 
with ConfigurableAgent system.

This script shows how to:
1. Set up agents with evaluation capabilities
2. Run single evaluations
3. Create and manage evaluation datasets
4. Run batch evaluations
5. Perform A/B testing
6. Use evaluation workflows
7. Analyze evaluation results

Prerequisites:
- Set OPENAI_API_KEY, LANGSMITH_API_KEY environment variables
- Install dependencies: pip install -r requirements.txt
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.configurable_agent import ConfigurableAgent
from evaluation.workflows import EvaluationWorkflow
from evaluation.evaluation_manager import EvaluationManager


def check_environment():
    """Check if required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment")
        return False
    
    print("✅ Environment variables are set")
    return True


def example_1_basic_evaluation():
    """Example 1: Basic single evaluation with an agent."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Single Evaluation")
    print("="*60)
    
    # Load agent with evaluation enabled
    config_path = "configs/examples/evaluation_enabled_agent.yml"
    agent = ConfigurableAgent(config_path)
    
    print(f"Agent loaded: {agent.config.agent.name}")
    print(f"Evaluation enabled: {agent.config.evaluation.enabled}")
    
    # Test query
    test_query = "What is the capital of France and what is it famous for?"
    
    print(f"\nTesting query: '{test_query}'")
    print("Running agent...")
    
    # Run agent (with automatic evaluation if configured)
    result = agent.run(test_query)
    
    print(f"\nAgent Response: {result['response']}")
    print(f"Response Time: {result.get('response_time', 0):.2f}s")
    
    # Check if automatic evaluation ran
    if "evaluation" in result:
        print("\n📊 Automatic Evaluation Results:")
        for evaluator_name, eval_result in result["evaluation"].items():
            if isinstance(eval_result, dict) and "score" in eval_result:
                score = eval_result["score"]
                reasoning = eval_result.get("reasoning", "No reasoning provided")
                print(f"  {evaluator_name}: {score:.2f}/1.0")
                print(f"    Reasoning: {reasoning[:100]}...")
    
    # Manual evaluation example
    print("\n📊 Manual Evaluation:")
    expected_output = {
        "answer": "Paris is the capital of France, famous for the Eiffel Tower, Louvre Museum, and rich culture."
    }
    
    manual_eval = agent.evaluate_single(test_query, expected_output)
    
    for evaluator_name, eval_result in manual_eval.items():
        if isinstance(eval_result, dict) and "score" in eval_result:
            score = eval_result["score"]
            print(f"  {evaluator_name}: {score:.2f}/1.0")


def example_2_dataset_creation_and_evaluation():
    """Example 2: Create dataset and run batch evaluation."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Dataset Creation and Batch Evaluation")
    print("="*60)
    
    # Load agent
    config_path = "configs/examples/evaluation_enabled_agent.yml"
    agent = ConfigurableAgent(config_path)
    
    # Create test dataset
    test_cases = [
        {
            "input": "What is 2 + 2?",
            "expected_output": {"answer": "4"},
            "category": "math"
        },
        {
            "input": "Who wrote Romeo and Juliet?",
            "expected_output": {"answer": "William Shakespeare"},
            "category": "literature"
        },
        {
            "input": "What is the largest planet in our solar system?",
            "expected_output": {"answer": "Jupiter"},
            "category": "science"
        },
        {
            "input": "Explain the concept of gravity",
            "expected_output": {"answer": "Gravity is a fundamental force that attracts objects with mass toward each other"},
            "category": "science"
        }
    ]
    
    print(f"Created test dataset with {len(test_cases)} cases")
    
    # Create evaluation workflow
    workflow = EvaluationWorkflow(agent)
    
    print("\nRunning batch evaluation...")
    batch_result = workflow.run_batch_evaluation(
        test_cases=test_cases,
        evaluator_names=["correctness", "helpfulness", "response_time"],
        max_workers=2,
        save_results=True
    )
    
    print(f"\n📊 Batch Evaluation Results:")
    print(f"  Total cases: {batch_result['total_cases']}")
    print(f"  Successful: {batch_result['successful_cases']}")
    print(f"  Failed: {batch_result['failed_cases']}")
    print(f"  Success rate: {batch_result['success_rate']:.1%}")
    print(f"  Duration: {batch_result['duration_seconds']:.1f}s")
    
    # Show summary metrics
    summary_metrics = batch_result['summary_metrics']
    if summary_metrics:
        print(f"\n📈 Summary Metrics:")
        for evaluator_name, metrics in summary_metrics.items():
            print(f"  {evaluator_name}:")
            print(f"    Average score: {metrics['average']:.3f}")
            print(f"    Min/Max: {metrics['min']:.3f} / {metrics['max']:.3f}")
            print(f"    Cases ≥ 0.7: {metrics['scores_above_threshold']['0.7']}/{metrics['count']}")


def example_3_ab_testing():
    """Example 3: A/B testing with different configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 3: A/B Testing Different Configurations")
    print("="*60)
    
    # Test cases for A/B testing
    ab_test_cases = [
        {"input": "Explain artificial intelligence", "category": "tech"},
        {"input": "What are the benefits of exercise?", "category": "health"},
        {"input": "How does photosynthesis work?", "category": "science"},
        {"input": "What is the meaning of life?", "category": "philosophy"}
    ]
    
    # Configuration variants to test
    # Note: In a real scenario, you'd have different config files
    # Here we'll simulate with the same config for demonstration
    variant_configs = {
        "variant_a": "configs/examples/evaluation_enabled_agent.yml",
        "variant_b": "configs/examples/research_agent.yml"  # Different agent as variant B
    }
    
    print(f"Testing {len(variant_configs)} variants with {len(ab_test_cases)} test cases")
    
    # Load base agent for workflow
    base_agent = ConfigurableAgent("configs/examples/evaluation_enabled_agent.yml")
    workflow = EvaluationWorkflow(base_agent)
    
    print("\nRunning A/B test...")
    ab_result = workflow.run_a_b_test(
        test_cases=ab_test_cases,
        variant_configs=variant_configs,
        evaluator_names=["correctness", "helpfulness"]
    )
    
    print(f"\n🔬 A/B Test Results:")
    print(f"  Test ID: {ab_result['test_id']}")
    print(f"  Duration: {ab_result['duration_seconds']:.1f}s")
    print(f"  Winner: {ab_result.get('winner', 'No clear winner')}")
    print(f"  Confidence: {ab_result.get('confidence', 'N/A')}")
    
    # Show variant comparison
    comparison = ab_result.get('comparison', {})
    if 'overall_scores' in comparison:
        print(f"\n📊 Variant Performance:")
        for variant_name, scores in comparison['overall_scores'].items():
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            print(f"  {variant_name}: {avg_score:.3f} average")
            for evaluator, score in scores.items():
                print(f"    {evaluator}: {score:.3f}")


def example_4_custom_evaluator():
    """Example 4: Creating and using a custom evaluator."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Evaluator")
    print("="*60)
    
    from evaluation.evaluators.base import BaseEvaluator
    
    class LengthEvaluator(BaseEvaluator):
        """Custom evaluator that scores based on response length."""
        
        def __init__(self, target_length: int = 200, tolerance: int = 50):
            super().__init__(
                name="response_length",
                description="Evaluates response length appropriateness"
            )
            self.target_length = target_length
            self.tolerance = tolerance
        
        def evaluate(self, inputs, outputs, reference_outputs=None, **kwargs):
            response = outputs.get("response", "")
            response_length = len(response)
            
            # Calculate score based on how close to target length
            diff = abs(response_length - self.target_length)
            if diff <= self.tolerance:
                score = 1.0
            elif diff <= self.tolerance * 2:
                score = 0.7
            elif diff <= self.tolerance * 3:
                score = 0.5
            else:
                score = 0.3
            
            reasoning = f"Response length: {response_length} chars (target: {self.target_length}±{self.tolerance})"
            
            return {
                "score": score,
                "reasoning": reasoning,
                "metadata": {
                    "response_length": response_length,
                    "target_length": self.target_length,
                    "difference": diff
                }
            }
    
    # Load agent and register custom evaluator
    agent = ConfigurableAgent("configs/examples/evaluation_enabled_agent.yml")
    
    if agent.evaluation_manager:
        custom_evaluator = LengthEvaluator(target_length=150, tolerance=30)
        agent.evaluation_manager.register_evaluator("response_length", custom_evaluator)
        print("✅ Registered custom length evaluator")
        
        # Test the custom evaluator
        test_query = "Briefly explain machine learning"
        result = agent.run(test_query)
        
        print(f"\nTesting custom evaluator with: '{test_query}'")
        print(f"Response length: {len(result['response'])} characters")
        
        # Manual evaluation with custom evaluator
        evaluation = agent.evaluation_manager.evaluate_single(
            input_data={"query": test_query},
            output_data=result,
            evaluator_names=["response_length"]
        )
        
        if "response_length" in evaluation:
            eval_result = evaluation["response_length"]
            print(f"Length evaluation score: {eval_result['score']:.2f}")
            print(f"Reasoning: {eval_result['reasoning']}")
    else:
        print("❌ Evaluation manager not available")


def example_5_metrics_analysis():
    """Example 5: Analyzing evaluation metrics over time."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Metrics Analysis")
    print("="*60)
    
    agent = ConfigurableAgent("configs/examples/evaluation_enabled_agent.yml")
    
    if not agent.evaluation_manager:
        print("❌ Evaluation manager not available")
        return
    
    # Run multiple evaluations to generate metrics
    test_queries = [
        "What is Python programming?",
        "Explain blockchain technology",
        "How do vaccines work?",
        "What is climate change?",
        "Describe quantum computing"
    ]
    
    print(f"Running {len(test_queries)} evaluations to generate metrics...")
    
    for i, query in enumerate(test_queries):
        print(f"  Running evaluation {i+1}/{len(test_queries)}")
        result = agent.run(query)
        
        # Manual evaluation for metrics
        agent.evaluation_manager.evaluate_single(
            input_data={"query": query},
            output_data=result
        )
        
        time.sleep(0.1)  # Small delay to show time-based metrics
    
    # Get metrics summary
    metrics_summary = agent.get_evaluation_metrics()
    
    print(f"\n📈 Evaluation Metrics Summary:")
    print(f"  Total evaluations: {metrics_summary.get('total_evaluations', 0)}")
    
    # Overall metrics
    overall_metrics = metrics_summary.get('overall_metrics', {})
    if overall_metrics:
        print(f"  Overall average score: {overall_metrics.get('average_score', 0):.3f}")
        print(f"  Score range: {overall_metrics.get('min_score', 0):.3f} - {overall_metrics.get('max_score', 0):.3f}")
        print(f"  Standard deviation: {overall_metrics.get('std_deviation', 0):.3f}")
    
    # Per-evaluator metrics
    evaluator_metrics = metrics_summary.get('evaluator_metrics', {})
    if evaluator_metrics:
        print(f"\n📊 Per-Evaluator Performance:")
        for evaluator_name, metrics in evaluator_metrics.items():
            print(f"  {evaluator_name}:")
            print(f"    Average: {metrics.get('average_score', 0):.3f}")
            print(f"    Count: {metrics.get('count', 0)}")
            print(f"    Distribution: {metrics.get('score_distribution', {})}")
    
    # Trends
    trends = metrics_summary.get('trends', {})
    if trends:
        print(f"\n📈 Performance Trends:")
        print(f"  Direction: {trends.get('direction', 'unknown')}")
        print(f"  Slope: {trends.get('slope', 0):.4f}")


def example_6_continuous_evaluation():
    """Example 6: Short continuous evaluation demo."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Continuous Evaluation (Short Demo)")
    print("="*60)
    
    agent = ConfigurableAgent("configs/examples/evaluation_enabled_agent.yml")
    workflow = EvaluationWorkflow(agent)
    
    # Short test cases for continuous evaluation
    continuous_test_cases = [
        {"input": "What is 5 + 3?"},
        {"input": "Name a programming language"},
        {"input": "What color is the sky?"}
    ]
    
    print("Running 3-minute continuous evaluation with 1-minute intervals...")
    print("(This will run 3 batches total)")
    
    try:
        continuous_result = workflow.run_continuous_evaluation(
            test_cases=continuous_test_cases,
            duration_hours=0.05,  # 3 minutes
            interval_minutes=1,    # 1 minute intervals
            evaluator_names=["correctness", "response_time"]
        )
        
        print(f"\n⏱️ Continuous Evaluation Results:")
        print(f"  Total batches: {continuous_result['total_batches']}")
        print(f"  Actual duration: {continuous_result.get('planned_duration_hours', 0) * 60:.1f} minutes")
        
        # Show trend analysis
        trend_analysis = continuous_result.get('trend_analysis', {})
        if trend_analysis:
            print(f"  Success rate trend: {trend_analysis.get('success_rate_trend', 'N/A')}")
            print(f"  Average success rate: {trend_analysis.get('average_success_rate', 0):.1%}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Continuous evaluation interrupted by user")


def main():
    """Run all evaluation examples."""
    print("🚀 ConfigurableAgent Evaluation Examples")
    print("=" * 60)
    
    if not check_environment():
        return
    
    try:
        # Run examples
        example_1_basic_evaluation()
        example_2_dataset_creation_and_evaluation()
        example_3_ab_testing()
        example_4_custom_evaluator()
        example_5_metrics_analysis()
        
        # Ask user if they want to run continuous evaluation
        print("\n" + "="*60)
        response = input("Run continuous evaluation demo? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            example_6_continuous_evaluation()
        
        print("\n✅ All evaluation examples completed successfully!")
        print("\nNext steps:")
        print("1. Set up your own evaluation datasets")
        print("2. Create custom evaluators for your specific use case")
        print("3. Set up scheduled evaluations for continuous monitoring")
        print("4. Use A/B testing to optimize your agent configurations")
        
    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()