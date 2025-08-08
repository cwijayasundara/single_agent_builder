"""
Built-in evaluators for common evaluation scenarios.
"""
import re
import time
from typing import Dict, Any, Optional, List
from .base import BaseEvaluator

try:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class CorrectnessEvaluator(BaseEvaluator):
    """Evaluates the correctness of agent responses."""
    
    def __init__(self, model: str = "openai:gpt-4.1-mini"):
        super().__init__(
            name="correctness",
            description="Evaluates whether the agent's response is factually correct and accurate"
        )
        self.model = model
        self.llm = None
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = init_chat_model(model=self.model, temperature=0.0)
            except Exception:
                pass
    
    def evaluate(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any], 
        reference_outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate correctness using LLM-as-judge or heuristics."""
        
        question = inputs.get("query") or inputs.get("input") or inputs.get("question", "")
        response = outputs.get("response") or outputs.get("output") or str(outputs)
        
        if self.llm and reference_outputs:
            return self._evaluate_with_llm(question, response, reference_outputs)
        else:
            return self._evaluate_heuristic(question, response, reference_outputs)
    
    def _evaluate_with_llm(self, question: str, response: str, reference: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to evaluate correctness."""
        reference_answer = reference.get("answer") or reference.get("response", "")
        
        prompt = f"""
        Question: {question}
        Reference Answer: {reference_answer}
        Agent Response: {response}
        
        Evaluate whether the agent's response is factually correct compared to the reference answer.
        Consider:
        1. Factual accuracy
        2. Completeness of information
        3. Logical consistency
        
        Provide a score from 0.0 to 1.0 (where 1.0 is perfectly correct) and explain your reasoning.
        Format your response as:
        SCORE: [score]
        REASONING: [detailed explanation]
        """
        
        try:
            messages = [SystemMessage(content="You are an expert evaluator assessing response correctness."),
                       HumanMessage(content=prompt)]
            result = self.llm.invoke(messages)
            content = result.content
            
            # Parse score and reasoning
            score_match = re.search(r'SCORE:\s*([0-9.]+)', content)
            reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.5
            reasoning = reasoning_match.group(1).strip() if reasoning_match else content
            
            return {
                "score": min(max(score, 0.0), 1.0),
                "reasoning": reasoning,
                "metadata": {"evaluator": "llm", "model": self.model}
            }
            
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"LLM evaluation failed: {str(e)}",
                "metadata": {"evaluator": "fallback", "error": str(e)}
            }
    
    def _evaluate_heuristic(self, question: str, response: str, reference: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Use heuristic evaluation when LLM is not available."""
        score = 0.5  # Default neutral score
        reasoning_parts = []
        
        # Basic quality checks
        if len(response.strip()) < 10:
            score -= 0.3
            reasoning_parts.append("Response is very short")
        
        if "I don't know" in response or "I'm not sure" in response:
            score -= 0.2
            reasoning_parts.append("Response indicates uncertainty")
        
        # Check for common error indicators
        error_indicators = ["error", "failed", "couldn't", "unable to", "sorry"]
        if any(indicator in response.lower() for indicator in error_indicators):
            score -= 0.2
            reasoning_parts.append("Response contains error indicators")
        
        # If reference is available, do basic comparison
        if reference:
            reference_answer = str(reference.get("answer", ""))
            if reference_answer and reference_answer.lower() in response.lower():
                score += 0.3
                reasoning_parts.append("Response contains reference information")
        
        # Ensure score is in valid range
        score = min(max(score, 0.0), 1.0)
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Heuristic evaluation completed"
        
        return {
            "score": score,
            "reasoning": reasoning,
            "metadata": {"evaluator": "heuristic", "checks_performed": len(reasoning_parts)}
        }


class HelpfulnessEvaluator(BaseEvaluator):
    """Evaluates how helpful the agent's response is to the user."""
    
    def __init__(self, model: str = "openai:gpt-4.1-mini"):
        super().__init__(
            name="helpfulness",
            description="Evaluates whether the agent's response is helpful and addresses the user's needs"
        )
        self.model = model
        self.llm = None
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = init_chat_model(model=self.model, temperature=0.0)
            except Exception:
                pass
    
    def evaluate(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate helpfulness of the response."""
        
        question = inputs.get("query") or inputs.get("input") or inputs.get("question", "")
        response = outputs.get("response") or outputs.get("output") or str(outputs)
        
        if self.llm:
            return self._evaluate_with_llm(question, response)
        else:
            return self._evaluate_heuristic(question, response)
    
    def _evaluate_with_llm(self, question: str, response: str) -> Dict[str, Any]:
        """Use LLM to evaluate helpfulness."""
        prompt = f"""
        User Question: {question}
        Agent Response: {response}
        
        Evaluate how helpful this response is to the user. Consider:
        1. Does it directly address the question?
        2. Is the information actionable and useful?
        3. Is it clear and easy to understand?
        4. Does it provide sufficient detail?
        5. Is the tone appropriate and professional?
        
        Provide a score from 0.0 to 1.0 (where 1.0 is extremely helpful) and explain your reasoning.
        Format your response as:
        SCORE: [score]
        REASONING: [detailed explanation]
        """
        
        try:
            messages = [SystemMessage(content="You are an expert evaluator assessing response helpfulness."),
                       HumanMessage(content=prompt)]
            result = self.llm.invoke(messages)
            content = result.content
            
            # Parse score and reasoning
            score_match = re.search(r'SCORE:\s*([0-9.]+)', content)
            reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 0.5
            reasoning = reasoning_match.group(1).strip() if reasoning_match else content
            
            return {
                "score": min(max(score, 0.0), 1.0),
                "reasoning": reasoning,
                "metadata": {"evaluator": "llm", "model": self.model}
            }
            
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"LLM evaluation failed: {str(e)}",
                "metadata": {"evaluator": "fallback", "error": str(e)}
            }
    
    def _evaluate_heuristic(self, question: str, response: str) -> Dict[str, Any]:
        """Use heuristic evaluation for helpfulness."""
        score = 0.5
        reasoning_parts = []
        
        # Length checks
        response_length = len(response.strip())
        if response_length < 20:
            score -= 0.3
            reasoning_parts.append("Response is too short to be helpful")
        elif response_length > 50:
            score += 0.1
            reasoning_parts.append("Response provides substantial information")
        
        # Structure and clarity
        if any(marker in response for marker in ["1.", "2.", "First", "Next", "Finally"]):
            score += 0.2
            reasoning_parts.append("Response is well-structured")
        
        # Actionability indicators
        action_words = ["should", "can", "try", "consider", "recommend", "suggest"]
        if any(word in response.lower() for word in action_words):
            score += 0.2
            reasoning_parts.append("Response provides actionable advice")
        
        # Unhelpful patterns
        unhelpful_patterns = ["I don't know", "I can't help", "not sure", "unclear"]
        if any(pattern in response.lower() for pattern in unhelpful_patterns):
            score -= 0.3
            reasoning_parts.append("Response indicates inability to help")
        
        score = min(max(score, 0.0), 1.0)
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Heuristic helpfulness evaluation"
        
        return {
            "score": score,
            "reasoning": reasoning,
            "metadata": {"evaluator": "heuristic", "response_length": response_length}
        }


class ResponseTimeEvaluator(BaseEvaluator):
    """Evaluates the response time performance of the agent."""
    
    def __init__(self, target_time: float = 5.0, max_time: float = 30.0):
        super().__init__(
            name="response_time",
            description="Evaluates the response time performance of the agent"
        )
        self.target_time = target_time  # Ideal response time in seconds
        self.max_time = max_time        # Maximum acceptable time in seconds
    
    def evaluate(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate response time performance."""
        
        # Get timing information
        if start_time and end_time:
            response_time = end_time - start_time
        elif "response_time" in outputs:
            response_time = outputs["response_time"]
        elif "metadata" in outputs and "response_time" in outputs["metadata"]:
            response_time = outputs["metadata"]["response_time"]
        else:
            return {
                "score": 0.0,
                "reasoning": "No timing information available",
                "metadata": {"evaluator": "response_time", "error": "missing_timing"}
            }
        
        # Calculate score based on response time
        if response_time <= self.target_time:
            score = 1.0
            reasoning = f"Excellent response time: {response_time:.2f}s (target: {self.target_time}s)"
        elif response_time <= self.max_time:
            # Linear decay from target to max time
            score = 1.0 - ((response_time - self.target_time) / (self.max_time - self.target_time)) * 0.7
            reasoning = f"Acceptable response time: {response_time:.2f}s (target: {self.target_time}s, max: {self.max_time}s)"
        else:
            score = 0.3  # Minimum score for very slow responses
            reasoning = f"Slow response time: {response_time:.2f}s (exceeds max: {self.max_time}s)"
        
        return {
            "score": min(max(score, 0.0), 1.0),
            "reasoning": reasoning,
            "metadata": {
                "evaluator": "response_time",
                "response_time": response_time,
                "target_time": self.target_time,
                "max_time": self.max_time
            }
        }


class ToolUsageEvaluator(BaseEvaluator):
    """Evaluates the effectiveness of tool usage by the agent."""
    
    def __init__(self):
        super().__init__(
            name="tool_usage",
            description="Evaluates how effectively the agent uses available tools"
        )
    
    def evaluate(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate tool usage effectiveness."""
        
        # Extract tool usage information with better fallbacks
        tool_results = outputs.get("tool_results", {})
        iteration_count = outputs.get("iteration_count", 0)
        messages = outputs.get("messages", [])
        
        # Get available tools from metadata if available
        metadata = outputs.get("metadata", {})
        tools_available = metadata.get("tools_available", 0)
        
        score = 0.5  # Base score
        reasoning_parts = []
        
        # Enhanced query analysis
        query = inputs.get("query", inputs.get("input", inputs.get("question", ""))).lower()
        
        # Check if tools were needed based on query content
        needs_search = any(indicator in query for indicator in 
                          ["search", "find", "lookup", "current", "recent", "latest", "news", 
                           "price", "weather", "stock", "what is", "who is"])
        needs_calculation = any(indicator in query for indicator in 
                               ["calculate", "compute", "math", "number", "count", "sum", 
                                "multiply", "divide", "equation", "+", "-", "*", "/"])
        needs_file_ops = any(indicator in query for indicator in 
                            ["file", "read", "write", "save", "load", "document", "text"])
        
        tools_used = len(tool_results) > 0
        tools_needed = needs_search or needs_calculation or needs_file_ops
        
        # Evaluate appropriateness of tool usage
        if tools_needed and tools_used:
            score += 0.3
            reasoning_parts.append("Used tools appropriately for the task")
            
            # Bonus for using relevant tools
            relevant_tools_used = 0
            if needs_search and any("search" in tool.lower() or "web" in tool.lower() for tool in tool_results.keys()):
                relevant_tools_used += 1
            if needs_calculation and any("calc" in tool.lower() or "math" in tool.lower() for tool in tool_results.keys()):
                relevant_tools_used += 1
            if needs_file_ops and any("file" in tool.lower() for tool in tool_results.keys()):
                relevant_tools_used += 1
            
            if relevant_tools_used > 0:
                score += 0.1
                reasoning_parts.append(f"Used {relevant_tools_used} relevant tool(s)")
                
        elif tools_needed and not tools_used:
            score -= 0.3
            reasoning_parts.append("Should have used tools but didn't")
        elif not tools_needed and not tools_used:
            score += 0.1
            reasoning_parts.append("Correctly avoided unnecessary tool usage")
        elif not tools_needed and tools_used:
            score -= 0.1
            reasoning_parts.append("Used tools when not necessary")
        
        # Evaluate iteration efficiency
        if iteration_count > 0:
            if iteration_count <= 2:
                score += 0.2
                reasoning_parts.append("Very efficient tool usage (≤2 iterations)")
            elif iteration_count <= 4:
                score += 0.1
                reasoning_parts.append("Efficient tool usage (≤4 iterations)")
            elif iteration_count > 7:
                score -= 0.2
                reasoning_parts.append("Many iterations suggest inefficient tool usage")
        
        # Analyze tool execution success
        successful_tools = 0
        failed_tools = 0
        
        for tool_name, tool_data in tool_results.items():
            if isinstance(tool_data, dict):
                if "error" in tool_data or tool_data.get("success") == False:
                    failed_tools += 1
                else:
                    successful_tools += 1
            else:
                # Assume success if we got data back
                successful_tools += 1
        
        # Factor in tool success rate
        if tools_used > 0:
            success_rate = successful_tools / (successful_tools + failed_tools) if (successful_tools + failed_tools) > 0 else 0
            if success_rate >= 0.8:
                score += 0.1
                reasoning_parts.append(f"High tool success rate ({success_rate:.1%})")
            elif success_rate < 0.5:
                score -= 0.2
                reasoning_parts.append(f"Low tool success rate ({success_rate:.1%})")
            
            if failed_tools > 0:
                reasoning_parts.append(f"Had {failed_tools} tool execution failures")
        
        # Ensure score is in valid range
        score = min(max(score, 0.0), 1.0)
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Tool usage evaluation completed"
        
        return {
            "score": score,
            "reasoning": reasoning,
            "metadata": {
                "evaluator": "tool_usage",
                "tools_used": list(tool_results.keys()),
                "tools_available": tools_available,
                "iteration_count": iteration_count,
                "successful_tools": successful_tools,
                "failed_tools": failed_tools,
                "tools_needed": tools_needed,
                "success_rate": successful_tools / max(1, successful_tools + failed_tools)
            }
        }