#!/usr/bin/env python3
"""
Simple evaluation demonstration for ConfigurableAgent with LangSmith integration.

This is a minimal example showing:
1. Basic agent setup with evaluation
2. Single evaluation run
3. Viewing evaluation results

Run this after setting up your environment variables:
- OPENAI_API_KEY
- LANGSMITH_API_KEY (optional, will fall back to heuristic evaluation)
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.configurable_agent import ConfigurableAgent


def main():
    print("🔍 Simple Evaluation Demo")
    print("=" * 40)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it in your .env file.")
        return
    
    print("✅ OpenAI API key found")
    
    # Check LangSmith API key
    if os.getenv("LANGSMITH_API_KEY"):
        print("✅ LangSmith API key found - full evaluation available")
    else:
        print("⚠️ LangSmith API key not found - using heuristic evaluation only")
    
    try:
        # Load agent with evaluation enabled
        print("\n📂 Loading evaluation-enabled agent...")
        agent = ConfigurableAgent("configs/examples/evaluation_enabled_agent.yml")
        
        print(f"Agent: {agent.config.agent.name}")
        print(f"Evaluation enabled: {agent.config.evaluation.enabled}")
        
        # Simple test query
        query = "What is the capital of Japan?"
        
        print(f"\n🤖 Testing query: '{query}'")
        print("Running agent...")
        
        # Run the agent
        result = agent.run(query)
        
        print(f"\n💬 Response: {result['response']}")
        print(f"⏱️ Response time: {result.get('response_time', 0):.2f}s")
        
        # Check for evaluation results
        if "evaluation" in result:
            print("\n📊 Automatic Evaluation Results:")
            for evaluator_name, eval_result in result["evaluation"].items():
                if isinstance(eval_result, dict) and "score" in eval_result:
                    score = eval_result["score"]
                    print(f"  {evaluator_name}: {score:.2f}/1.0")
                    
                    # Show reasoning if available
                    reasoning = eval_result.get("reasoning", "")
                    if reasoning:
                        print(f"    → {reasoning[:80]}...")
        else:
            print("\n⚠️ No automatic evaluation results (check auto_evaluate setting)")
        
        # Manual evaluation example
        print("\n🔍 Running manual evaluation...")
        expected_answer = {"answer": "Tokyo is the capital of Japan"}
        
        manual_eval = agent.evaluate_single(query, expected_answer)
        
        print("Manual evaluation results:")
        for evaluator_name, eval_result in manual_eval.items():
            if isinstance(eval_result, dict) and "score" in eval_result:
                score = eval_result["score"]
                print(f"  {evaluator_name}: {score:.2f}/1.0")
        
        # Show evaluation metrics summary
        print("\n📈 Getting evaluation metrics...")
        metrics = agent.get_evaluation_metrics()
        
        if metrics.get("total_evaluations", 0) > 0:
            print(f"Total evaluations run: {metrics['total_evaluations']}")
            
            overall = metrics.get("overall_metrics", {})
            if overall:
                print(f"Overall average score: {overall.get('average_score', 0):.3f}")
        else:
            print("No metrics available yet")
        
        print("\n✅ Evaluation demo completed successfully!")
        print("\nTry running 'python examples/evaluation_examples.py' for more comprehensive examples.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()