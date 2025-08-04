"""
Memory management integration with LangMem for configurable agents.
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage
import hashlib

from ..core.config_loader import MemoryConfig
from ..custom_logging import get_logger

# Get logger for this module
logger = get_logger(__name__, "memory")


class MemoryManager:
    """Manages agent memory using LangMem integration."""
    
    def __init__(self, memory_config: MemoryConfig):
        self.config = memory_config
        self.semantic_memory = {}
        self.episodic_memory = []
        self.procedural_memory = {}
        self.memory_store = {}
        
        logger.info(
            "Initializing memory manager",
            extra={
                "enabled": memory_config.enabled,
                "backend": memory_config.storage.backend,
                "semantic_enabled": memory_config.types.semantic,
                "episodic_enabled": memory_config.types.episodic,
                "procedural_enabled": memory_config.types.procedural
            }
        )
        
        # Initialize based on configuration
        self._initialize_memory_backend()
    
    def _initialize_memory_backend(self):
        """Initialize memory backend based on configuration."""
        backend = self.config.storage.backend
        logger.debug(f"Initializing memory backend: {backend}")
        
        if backend == "memory":
            # In-memory storage (default)
            logger.debug("Using in-memory storage backend")
        elif backend == "postgres":
            # Initialize PostgreSQL connection
            # This would integrate with actual LangMem PostgreSQL backend
            logger.info("Initializing PostgreSQL memory backend")
            # TODO: Implement PostgreSQL backend
        elif backend == "redis":
            # Initialize Redis connection
            # This would integrate with actual LangMem Redis backend
            logger.info("Initializing Redis memory backend")
            # TODO: Implement Redis backend
        else:
            logger.warning(
                "Unknown memory backend, falling back to in-memory storage",
                extra={"backend": backend, "fallback": "memory"}
            )
    
    def store_interaction(self, messages: List[BaseMessage], response: BaseMessage):
        """Store an interaction in memory."""
        if not self.config.enabled:
            logger.debug("Memory storage disabled, skipping interaction storage")
            return
        
        try:
            interaction_id = self._generate_interaction_id(messages, response)
            timestamp = datetime.now()
            
            logger.debug(
                "Storing interaction in memory",
                extra={
                    "interaction_id": interaction_id,
                    "messages_count": len(messages),
                    "has_response": bool(response)
                }
            )
            
            interaction = {
                "id": interaction_id,
                "timestamp": timestamp.isoformat(),
                "messages": [self._message_to_dict(msg) for msg in messages],
                "response": self._message_to_dict(response),
                "metadata": {}
            }
            
            # Store in episodic memory
            if self.config.types.episodic:
                self.episodic_memory.append(interaction)
                self._cleanup_old_episodes()
                logger.log_memory_operation("store", "episodic", success=True)
            
            # Extract semantic information
            if self.config.types.semantic:
                self._extract_semantic_info(messages, response)
                logger.log_memory_operation("extract", "semantic", success=True)
            
            # Update procedural memory
            if self.config.types.procedural:
                self._update_procedural_memory(messages, response)
                logger.log_memory_operation("update", "procedural", success=True)
            
            logger.debug(
                "Interaction stored successfully",
                extra={
                    "interaction_id": interaction_id,
                    "episodic_count": len(self.episodic_memory),
                    "semantic_count": len(self.semantic_memory),
                    "procedural_count": len(self.procedural_memory)
                }
            )
            
        except Exception as e:
            logger.error(
                "Failed to store interaction in memory",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "messages_count": len(messages) if messages else 0
                },
                exc_info=True
            )
            logger.log_memory_operation("store", "all", success=False)
    
    def retrieve_memory(self, query: str, memory_type: str = "all") -> Dict[str, Any]:
        """Retrieve relevant memory based on query."""
        if not self.config.enabled:
            return {}
        
        relevant_memory = {}
        
        if memory_type in ["all", "semantic"] and self.config.types.semantic:
            relevant_memory["semantic"] = self._search_semantic_memory(query)
        
        if memory_type in ["all", "episodic"] and self.config.types.episodic:
            relevant_memory["episodic"] = self._search_episodic_memory(query)
        
        if memory_type in ["all", "procedural"] and self.config.types.procedural:
            relevant_memory["procedural"] = self._get_procedural_memory(query)
        
        return relevant_memory
    
    def get_relevant_context(self, query: str, max_items: int = 5) -> str:
        """Get relevant context as a formatted string."""
        memory = self.retrieve_memory(query)
        context_parts = []
        
        # Add semantic context
        if "semantic" in memory and memory["semantic"]:
            facts = memory["semantic"][:max_items]
            context_parts.append("Relevant facts: " + "; ".join(facts))
        
        # Add episodic context
        if "episodic" in memory and memory["episodic"]:
            episodes = memory["episodic"][:max_items]
            episode_summaries = [ep.get("summary", "Previous interaction") for ep in episodes]
            context_parts.append("Previous interactions: " + "; ".join(episode_summaries))
        
        # Add procedural context
        if "procedural" in memory and memory["procedural"]:
            procedures = memory["procedural"]
            context_parts.append("Learned procedures: " + "; ".join(procedures.keys()))
        
        return " | ".join(context_parts) if context_parts else ""
    
    def store_memory(self, content: BaseMessage):
        """Store individual memory item."""
        if not self.config.enabled:
            return
        
        memory_id = self._generate_memory_id(content)
        
        memory_item = {
            "id": memory_id,
            "content": self._message_to_dict(content),
            "timestamp": datetime.now().isoformat(),
            "type": "manual_storage"
        }
        
        self.memory_store[memory_id] = memory_item
    
    def _extract_semantic_info(self, messages: List[BaseMessage], response: BaseMessage):
        """Extract semantic information from interaction."""
        # Simple semantic extraction (in real implementation, this would use NLP)
        content = " ".join([msg.content for msg in messages] + [response.content])
        
        # Look for facts patterns (this is simplified)
        if "is" in content.lower():
            sentences = content.split(".")
            for sentence in sentences:
                if " is " in sentence.lower():
                    fact = sentence.strip()
                    fact_id = hashlib.md5(fact.encode()).hexdigest()[:8]
                    self.semantic_memory[fact_id] = {
                        "fact": fact,
                        "confidence": 0.8,
                        "timestamp": datetime.now().isoformat()
                    }
    
    def _search_semantic_memory(self, query: str) -> List[str]:
        """Search semantic memory for relevant facts."""
        query_words = set(query.lower().split())
        relevant_facts = []
        
        for fact_data in self.semantic_memory.values():
            fact_words = set(fact_data["fact"].lower().split())
            # Simple word overlap scoring
            overlap = len(query_words.intersection(fact_words))
            if overlap > 0:
                relevant_facts.append((fact_data["fact"], overlap))
        
        # Sort by relevance and return top facts
        relevant_facts.sort(key=lambda x: x[1], reverse=True)
        return [fact[0] for fact in relevant_facts[:5]]
    
    def _search_episodic_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search episodic memory for relevant interactions."""
        query_words = set(query.lower().split())
        relevant_episodes = []
        
        for episode in self.episodic_memory:
            # Search in all messages of the episode
            episode_text = " ".join([
                msg["content"] for msg in episode["messages"]
            ] + [episode["response"]["content"]])
            
            episode_words = set(episode_text.lower().split())
            overlap = len(query_words.intersection(episode_words))
            
            if overlap > 0:
                relevant_episodes.append((episode, overlap))
        
        # Sort by relevance and return top episodes
        relevant_episodes.sort(key=lambda x: x[1], reverse=True)
        return [ep[0] for ep in relevant_episodes[:3]]
    
    def _update_procedural_memory(self, messages: List[BaseMessage], response: BaseMessage):
        """Update procedural memory based on successful patterns."""
        # Simple pattern detection (in real implementation, this would be more sophisticated)
        if "successfully" in response.content.lower() or "completed" in response.content.lower():
            # Extract the pattern that led to success
            if messages:
                pattern = messages[-1].content[:100]  # First 100 chars as pattern
                pattern_id = hashlib.md5(pattern.encode()).hexdigest()[:8]
                
                if pattern_id in self.procedural_memory:
                    self.procedural_memory[pattern_id]["success_count"] += 1
                else:
                    self.procedural_memory[pattern_id] = {
                        "pattern": pattern,
                        "success_count": 1,
                        "last_used": datetime.now().isoformat()
                    }
    
    def _get_procedural_memory(self, query: str) -> Dict[str, Any]:
        """Get relevant procedural memory."""
        # Return most successful patterns
        sorted_procedures = sorted(
            self.procedural_memory.items(),
            key=lambda x: x[1]["success_count"],
            reverse=True
        )
        
        return {
            proc_id: proc_data["pattern"] 
            for proc_id, proc_data in sorted_procedures[:3]
        }
    
    def _cleanup_old_episodes(self):
        """Clean up old episodic memories based on retention policy."""
        if len(self.episodic_memory) > self.config.settings.max_memory_size:
            # Remove oldest episodes
            self.episodic_memory = self.episodic_memory[-self.config.settings.max_memory_size:]
        
        # Remove episodes older than retention period
        cutoff_date = datetime.now() - timedelta(days=self.config.settings.retention_days)
        self.episodic_memory = [
            episode for episode in self.episodic_memory
            if datetime.fromisoformat(episode["timestamp"]) > cutoff_date
        ]
    
    def _generate_interaction_id(self, messages: List[BaseMessage], response: BaseMessage) -> str:
        """Generate unique ID for interaction."""
        content = str(messages) + str(response) + str(datetime.now())
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_memory_id(self, content: BaseMessage) -> str:
        """Generate unique ID for memory item."""
        content_str = str(content) + str(datetime.now())
        return hashlib.md5(content_str.encode()).hexdigest()[:12]
    
    def _message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "type": type(message).__name__,
            "content": message.content,
            "additional_kwargs": getattr(message, "additional_kwargs", {})
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "memory_enabled": self.config.enabled,
            "semantic_facts": len(self.semantic_memory),
            "episodic_interactions": len(self.episodic_memory),
            "procedural_patterns": len(self.procedural_memory),
            "manual_memories": len(self.memory_store),
            "retention_days": self.config.settings.retention_days,
            "max_memory_size": self.config.settings.max_memory_size
        }
    
    def clear_memory(self):
        """Clear all memory."""
        self.semantic_memory.clear()
        self.episodic_memory.clear()
        self.procedural_memory.clear()
        self.memory_store.clear()
    
    def export_history(self, format_type: str = "json") -> str:
        """Export memory history."""
        history = {
            "semantic_memory": self.semantic_memory,
            "episodic_memory": self.episodic_memory,
            "procedural_memory": self.procedural_memory,
            "memory_store": self.memory_store,
            "export_timestamp": datetime.now().isoformat()
        }
        
        if format_type.lower() == "json":
            return json.dumps(history, indent=2)
        else:
            return str(history)
    
    def import_history(self, history_data: str, format_type: str = "json"):
        """Import memory history."""
        if format_type.lower() == "json":
            data = json.loads(history_data)
            self.semantic_memory = data.get("semantic_memory", {})
            self.episodic_memory = data.get("episodic_memory", [])
            self.procedural_memory = data.get("procedural_memory", {})
            self.memory_store = data.get("memory_store", {})