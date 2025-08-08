# Serper API Search Setup Guide

This guide explains how to set up Serper API for web search functionality in your agent system.

## Current Search Capabilities

**Without Serper API Key:**
- ❌ Web search unavailable - requires API key

**With Serper API Key:**
- 🚀 **Real Google search results** with high quality
- 🚀 **Answer boxes and featured snippets** for direct answers
- 🚀 **Knowledge graph information** for entity queries  
- 🚀 **Comprehensive web results** from Google's index
- 🚀 **Generic search** - works for any query (news, prices, facts, etc.)

## Why Serper API?

- **Real Google Results**: Uses actual Google search, not a separate index
- **Generous Free Tier**: 2,500 free searches for new users
- **Cost Effective**: Only $0.001 per search after free tier
- **Rich Results**: Includes answer boxes, snippets, and knowledge graph data
- **Fast & Reliable**: Average response time under 1 second
- **Easy Setup**: Just one API key needed

## Setup Steps

### 1. Get Your Serper API Key

1. Visit [serper.dev](https://serper.dev/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. You'll receive **2,500 free searches** to start

### 2. Configure Environment Variable

Add to your `.env` file:
```bash
SERPER_API_KEY=your-serper-api-key-here
```

Or set as environment variable:
```bash
export SERPER_API_KEY=your-serper-api-key-here
```

### 3. Test Your Setup

```python
from src.tools.tool_registry import ToolRegistry

registry = ToolRegistry()
web_search_tool = registry.get_tool('web_search')

# Test search
result = web_search_tool.invoke({'query': 'latest artificial intelligence news'})
print(result)
```

Look for `"via Serper API"` in the search results to confirm it's working.

## Search Result Types

The Serper integration provides rich, structured results:

### Answer Boxes
Direct answers to factual questions:
```
Answer: The capital of France is Paris.
```

### Featured Snippets
Highlighted content from web pages:
```
Featured Snippet: Python is a high-level programming language...
```

### Knowledge Graph
Information about entities:
```
About: Python is an interpreted, high-level programming language...
```

### Web Results
Standard search results with titles and snippets:
```
• Python.org - Welcome to Python.org: The official home of the Python...
• Learn Python - Tutorial: Python is a powerful programming language...
```

## Example Search Queries

The generic web search works for any type of query:

```python
# Current events
web_search_tool.invoke({'query': 'latest AI developments 2025'})

# Financial information
web_search_tool.invoke({'query': 'bitcoin price today'})

# Scientific topics
web_search_tool.invoke({'query': 'NASA Mars mission updates'})

# Technology news  
web_search_tool.invoke({'query': 'iPhone 16 release date'})

# General knowledge
web_search_tool.invoke({'query': 'how does photosynthesis work'})
```

## Usage & Pricing

### Free Tier
- **2,500 searches** for new accounts
- Perfect for development and testing
- No time limits or expiration

### Paid Usage
- **$0.001 per search** (very affordable)
- **$1 = 1,000 searches**
- **$10 = 10,000 searches**
- Pay only for what you use

### Usage Monitoring
- Track usage in your Serper dashboard
- Set up usage alerts
- View detailed analytics

## Error Handling

The system provides clear error messages:

```
Web search unavailable: Please set SERPER_API_KEY environment variable. 
Get your free API key at https://serper.dev (2,500 free searches).
```

Common issues:
- **Missing API key**: Set `SERPER_API_KEY` environment variable
- **Invalid API key**: Check your key in the Serper dashboard
- **Quota exceeded**: Upgrade your plan or wait for reset
- **Network issues**: Check internet connection

## Security Best Practices

- ✅ Keep API keys in environment variables
- ✅ Add `.env` to `.gitignore`
- ✅ Use different keys for development/production
- ✅ Monitor usage to prevent unexpected charges
- ❌ Never commit API keys to version control

## Production Deployment

For production environments:

1. **Set environment variable**:
   ```bash
   export SERPER_API_KEY=your-production-api-key
   ```

2. **Monitor usage** via Serper dashboard

3. **Set usage alerts** to avoid surprises

4. **Consider usage patterns**:
   - Average queries per user/day
   - Peak usage times
   - Caching strategies if needed

## Troubleshooting

### API Key Not Working
```bash
# Test your API key directly:
curl -X POST https://google.serper.dev/search \
  -H 'X-API-KEY: your-api-key-here' \
  -H 'Content-Type: application/json' \
  -d '{"q": "test query"}'
```

### Environment Variable Issues
```python
import os
print(f"SERPER_API_KEY set: {bool(os.getenv('SERPER_API_KEY'))}")
```

### Restart Required
After setting environment variables, restart your application.

## Support & Resources

- **Serper Documentation**: [docs.serper.dev](https://docs.serper.dev)
- **Support**: Contact through Serper dashboard
- **Status Page**: [status.serper.dev](https://status.serper.dev)
- **Pricing Calculator**: Available on serper.dev

## Migration from Other APIs

If migrating from Google Custom Search or other APIs:

1. **Remove old environment variables**
2. **Set `SERPER_API_KEY`**  
3. **Test functionality**
4. **Monitor usage patterns**

The Serper integration provides superior results with better pricing and easier setup compared to Google's official Custom Search API.