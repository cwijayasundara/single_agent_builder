"""
Memory management integration with LangMem for configurable agents.
"""
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import hashlib

from ..core.config_loader import MemoryConfig
from ..custom_logging import get_logger

# Get logger for this module
logger = get_logger(__name__, "memory")

# LangMem integration with graceful fallback
try:
    from langmem import Client, MemoryManager as LangMemManager
    LANGMEM_AVAILABLE = True
    logger.info("LangMem SDK available for memory management")
except ImportError as e:
    LANGMEM_AVAILABLE = False
    logger.warning(f"LangMem SDK not available, falling back to basic memory: {e}")
    # Create placeholder classes for development without LangMem
    class Client:
        def __init__(self, *args, **kwargs):
            pass
    class LangMemManager:
        def __init__(self, *args, **kwargs):
            pass


class MemoryManager:
    """Manages agent memory using LangMem integration."""
    
    def __init__(self, memory_config: MemoryConfig):
        self.config = memory_config
        self.langmem_client = None
        self.langmem_manager = None
        
        # Fallback storage for when LangMem is not available
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
                "procedural_enabled": memory_config.types.procedural,
                "langmem_available": LANGMEM_AVAILABLE
            }
        )
        
        # Initialize based on configuration
        self._initialize_memory_backend()
    
    def _initialize_memory_backend(self):
        """Initialize memory backend based on configuration."""
        backend = self.config.storage.backend
        logger.debug(f"Initializing memory backend: {backend}")
        
        if LANGMEM_AVAILABLE and self.config.enabled:
            try:
                # Initialize LangMem client based on backend
                if backend == "memory":
                    # In-memory storage using LangMem
                    logger.debug("Initializing LangMem with in-memory storage")
                    self.langmem_client = Client()
                    self.langmem_manager = LangMemManager(self.langmem_client)
                    
                elif backend == "postgres":
                    # PostgreSQL backend with LangMem
                    logger.info("Initializing LangMem with PostgreSQL backend")
                    connection_string = self.config.storage.connection_string
                    if connection_string:
                        self.langmem_client = Client(
                            store_type="postgres",
                            connection_string=connection_string
                        )
                    else:
                        logger.warning("PostgreSQL connection string not provided, falling back to in-memory")
                        self.langmem_client = Client()
                    self.langmem_manager = LangMemManager(self.langmem_client)
                    
                elif backend == "redis":
                    # Redis backend with LangMem
                    logger.info("Initializing LangMem with Redis backend")
                    connection_string = self.config.storage.connection_string
                    if connection_string:
                        self.langmem_client = Client(
                            store_type="redis",
                            connection_string=connection_string
                        )
                    else:
                        logger.warning("Redis connection string not provided, falling back to in-memory")
                        self.langmem_client = Client()
                    self.langmem_manager = LangMemManager(self.langmem_client)
                    
                else:
                    logger.warning(
                        "Unknown memory backend, using LangMem with in-memory storage",
                        extra={"backend": backend, "fallback": "memory"}
                    )
                    self.langmem_client = Client()
                    self.langmem_manager = LangMemManager(self.langmem_client)
                
                logger.info(
                    "LangMem memory backend initialized successfully",
                    extra={
                        "backend": backend,
                        "has_client": bool(self.langmem_client),
                        "has_manager": bool(self.langmem_manager)
                    }
                )
                
            except Exception as e:
                logger.error(
                    "Failed to initialize LangMem backend, falling back to basic memory",
                    extra={"backend": backend, "error": str(e)},
                    exc_info=True
                )
                self.langmem_client = None
                self.langmem_manager = None
        else:
            if not LANGMEM_AVAILABLE:
                logger.warning("LangMem not available, using fallback memory implementation")
            else:
                logger.debug("Memory disabled in configuration")
    
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
                    "has_response": bool(response),
                    "using_langmem": bool(self.langmem_manager)
                }
            )
            
            # Use LangMem if available
            if self.langmem_manager:
                try:
                    # Store interaction using LangMem
                    conversation = messages + [response]
                    
                    # Store episodic memory (conversation history)
                    if self.config.types.episodic:
                        self.langmem_manager.create_memory(
                            content=self._format_conversation_for_storage(conversation),
                            memory_type="episodic",
                            metadata={
                                "interaction_id": interaction_id,
                                "timestamp": timestamp.isoformat(),
                                "message_count": len(messages)
                            }
                        )
                        logger.log_memory_operation("store", "episodic", success=True)
                    
                    # Extract and store semantic information
                    if self.config.types.semantic:
                        semantic_content = self._extract_semantic_content(messages, response)
                        if semantic_content:
                            self.langmem_manager.create_memory(
                                content=semantic_content,
                                memory_type="semantic",
                                metadata={
                                    "interaction_id": interaction_id,
                                    "timestamp": timestamp.isoformat(),
                                    "source": "conversation_extraction"
                                }
                            )
                        logger.log_memory_operation("extract", "semantic", success=True)
                    
                    # Update procedural memory (patterns and learnings)
                    if self.config.types.procedural:
                        procedural_insights = self._extract_procedural_insights(messages, response)
                        if procedural_insights:
                            self.langmem_manager.create_memory(
                                content=procedural_insights,
                                memory_type="procedural",
                                metadata={
                                    "interaction_id": interaction_id,
                                    "timestamp": timestamp.isoformat(),
                                    "pattern_type": "successful_interaction"
                                }
                            )
                        logger.log_memory_operation("update", "procedural", success=True)
                    
                    logger.debug("Interaction stored successfully using LangMem")
                    
                except Exception as langmem_error:
                    logger.warning(
                        "LangMem storage failed, falling back to basic memory",
                        extra={"error": str(langmem_error)},
                        exc_info=True
                    )
                    # Fall back to basic memory storage
                    self._store_interaction_fallback(messages, response, interaction_id, timestamp)
            else:
                # Use fallback memory storage
                self._store_interaction_fallback(messages, response, interaction_id, timestamp)
            
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
        
        # Use LangMem if available
        if self.langmem_manager:
            try:
                if memory_type in ["all", "semantic"] and self.config.types.semantic:
                    semantic_results = self.langmem_manager.search_memories(
                        query=query,
                        memory_type="semantic",
                        limit=5
                    )
                    relevant_memory["semantic"] = [result.content for result in semantic_results]
                
                if memory_type in ["all", "episodic"] and self.config.types.episodic:
                    episodic_results = self.langmem_manager.search_memories(
                        query=query,
                        memory_type="episodic",
                        limit=3
                    )
                    relevant_memory["episodic"] = [
                        {
                            "content": result.content,
                            "timestamp": result.metadata.get("timestamp"),
                            "interaction_id": result.metadata.get("interaction_id")
                        }
                        for result in episodic_results
                    ]
                
                if memory_type in ["all", "procedural"] and self.config.types.procedural:
                    procedural_results = self.langmem_manager.search_memories(
                        query=query,
                        memory_type="procedural",
                        limit=3
                    )
                    relevant_memory["procedural"] = {
                        f"pattern_{i}": result.content 
                        for i, result in enumerate(procedural_results)
                    }
                
                logger.debug(
                    "Memory retrieved using LangMem",
                    extra={
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "memory_types": list(relevant_memory.keys()),
                        "total_results": sum(len(v) if isinstance(v, list) else len(v) if isinstance(v, dict) else 1 for v in relevant_memory.values())
                    }
                )
                
            except Exception as langmem_error:
                logger.warning(
                    "LangMem retrieval failed, falling back to basic memory",
                    extra={"error": str(langmem_error)},
                    exc_info=True
                )
                # Fall back to basic memory retrieval
                relevant_memory = self._retrieve_memory_fallback(query, memory_type)
        else:
            # Use fallback memory retrieval
            relevant_memory = self._retrieve_memory_fallback(query, memory_type)
        
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
    
    # LangMem integration helper methods
    def _format_conversation_for_storage(self, conversation: List[BaseMessage]) -> str:
        """Format conversation for storage in LangMem."""
        formatted_parts = []
        for msg in conversation:
            if hasattr(msg, 'type'):
                msg_type = msg.type
            else:
                msg_type = type(msg).__name__.replace('Message', '').lower()
            formatted_parts.append(f"{msg_type.upper()}: {msg.content}")
        return "\n".join(formatted_parts)
    
    def _extract_semantic_content(self, messages: List[BaseMessage], response: BaseMessage) -> str:
        """Extract semantic facts and knowledge from conversation."""
        content = " ".join([msg.content for msg in messages] + [response.content])
        
        # Look for factual statements, definitions, and key information
        semantic_indicators = [
            " is ", " are ", " was ", " were ", " means ", " defines ", 
            " represents ", " indicates ", " shows ", " demonstrates ",
            " equals ", " refers to ", " known as "
        ]
        
        sentences = content.split(".")
        semantic_facts = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in semantic_indicators):
                if len(sentence) > 10 and len(sentence) < 200:  # Filter reasonable length
                    semantic_facts.append(sentence)
        
        return " | ".join(semantic_facts) if semantic_facts else None
    
    def _extract_procedural_insights(self, messages: List[BaseMessage], response: BaseMessage) -> str:
        """Extract procedural knowledge and patterns from successful interactions."""
        # Look for successful patterns, procedures, and learned behaviors
        response_content = response.content.lower()
        
        success_indicators = [
            "successfully", "completed", "finished", "done", "solved", 
            "worked", "correct", "right", "achieved", "accomplished"
        ]
        
        if any(indicator in response_content for indicator in success_indicators):
            # Extract the query pattern that led to success
            if messages:
                query_pattern = messages[-1].content[:150]  # Keep reasonable length
                return f"Successful pattern: {query_pattern} -> {response.content[:100]}"
        
        return None
    
    # Fallback methods for when LangMem is not available
    def _store_interaction_fallback(self, messages: List[BaseMessage], response: BaseMessage, 
                                  interaction_id: str, timestamp: datetime):
        """Store interaction using fallback memory implementation."""
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
            self._extract_semantic_info_fallback(messages, response)
            logger.log_memory_operation("extract", "semantic", success=True)
        
        # Update procedural memory
        if self.config.types.procedural:
            self._update_procedural_memory_fallback(messages, response)
            logger.log_memory_operation("update", "procedural", success=True)
    
    def _retrieve_memory_fallback(self, query: str, memory_type: str = "all") -> Dict[str, Any]:
        """Retrieve memory using fallback implementation."""
        relevant_memory = {}
        
        if memory_type in ["all", "semantic"] and self.config.types.semantic:
            relevant_memory["semantic"] = self._search_semantic_memory(query)
        
        if memory_type in ["all", "episodic"] and self.config.types.episodic:
            relevant_memory["episodic"] = self._search_episodic_memory(query)
        
        if memory_type in ["all", "procedural"] and self.config.types.procedural:
            relevant_memory["procedural"] = self._get_procedural_memory()
        
        return relevant_memory
    
    def _extract_semantic_info_fallback(self, messages: List[BaseMessage], response: BaseMessage):
        """Extract semantic information from interaction (fallback implementation)."""
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
    
    def _update_procedural_memory_fallback(self, messages: List[BaseMessage], response: BaseMessage):
        """Update procedural memory based on successful patterns (fallback implementation)."""
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
    
    def _get_procedural_memory(self) -> Dict[str, Any]:
        """Get relevant procedural memory (fallback implementation)."""
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
        stats = {
            "memory_enabled": self.config.enabled,
            "langmem_available": LANGMEM_AVAILABLE,
            "using_langmem": bool(self.langmem_manager),
            "backend": self.config.storage.backend,
            "retention_days": self.config.settings.retention_days,
            "max_memory_size": self.config.settings.max_memory_size,
            "memory_types": {
                "semantic": self.config.types.semantic,
                "episodic": self.config.types.episodic,
                "procedural": self.config.types.procedural
            }
        }
        
        # Add LangMem-specific stats if available
        if self.langmem_manager:
            try:
                # Try to get LangMem statistics
                stats.update({
                    "semantic_memories": "available_via_langmem",
                    "episodic_memories": "available_via_langmem", 
                    "procedural_memories": "available_via_langmem"
                })
            except Exception as e:
                logger.warning(f"Could not retrieve LangMem stats: {e}")
                stats.update({
                    "langmem_error": str(e)
                })
        else:
            # Use fallback memory stats
            stats.update({
                "semantic_facts": len(self.semantic_memory),
                "episodic_interactions": len(self.episodic_memory),
                "procedural_patterns": len(self.procedural_memory),
                "manual_memories": len(self.memory_store)
            })
        
        return stats
    
    def clear_memory(self):
        """Clear all memory."""
        if self.langmem_manager:
            try:
                # Clear LangMem storage (implementation depends on LangMem API)
                logger.warning("LangMem clear_memory not implemented - would need to clear via LangMem API")
            except Exception as e:
                logger.error(f"Failed to clear LangMem memory: {e}")
        
        # Clear fallback memory
        self.semantic_memory.clear()
        self.episodic_memory.clear()
        self.procedural_memory.clear()
        self.memory_store.clear()
        
        logger.info("Memory cleared")
    
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