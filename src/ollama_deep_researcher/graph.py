import json

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from ollama_deep_researcher.configuration import Configuration, SearchAPI
from ollama_deep_researcher.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search, duckduckgo_search, searxng_search, strip_thinking_tokens, get_config_value
from ollama_deep_researcher.state import SummaryState, SummaryStateInput, SummaryStateOutput
from ollama_deep_researcher.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, get_current_date
from ollama_deep_researcher.lmstudio import ChatLMStudio

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """LangGraph node that generates a search query based on the research topic.
    
    Uses an LLM to create an optimized search query for web research based on
    the user's research topic. Supports both LMStudio and Ollama as LLM providers.
    
    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    else: # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Generate a query for web search:")]
    )
    
    # Get the content
    content = result.content

    # Parse the JSON response and get the query
    try:
        # Print raw content for debugging
        print(f"Raw content from LLM: {content[:150]}...")
        
        # Try to parse as JSON
        query = json.loads(content)
        print(f"Successfully parsed JSON: {query}")
        
        # Extract the query string, not the whole JSON
        search_query = query.get('query')
        
        # Also get the keywords if available
        seo_keywords = query.get('seo_keywords', [])
        
        # Validate the search query
        if not search_query or not isinstance(search_query, str) or len(search_query.strip()) < 3:
            raise ValueError(f"Invalid search query: {search_query}")
            
        print(f"Extracted search query: {search_query}")
        
        # Return both the search query and SEO keywords
        return {
            "search_query": search_query,
            "seo_keywords": seo_keywords
        }
        
    except Exception as e:
        # If parsing fails or the key is not found, use a fallback query
        print(f"Error parsing LLM response: {str(e)}")
        
        # Try to clean up the content if needed
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
        
        # Create a fallback query based on the research topic
        fallback_query = f"Tell me more about {state.research_topic}"
        print(f"Using fallback query: {fallback_query}")
        
        return {"search_query": fallback_query}

def web_research(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs web research using the generated search query.
    
    Executes a web search using the configured search API (tavily, perplexity, 
    duckduckgo, or searxng) and formats the results for further processing.
    
    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings
        
    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """

    try:
        # Configure
        configurable = Configuration.from_runnable_config(config)

        # Get the search API
        search_api = get_config_value(configurable.search_api)

        # Print the search query for debugging
        print(f"Searching for: '{state.search_query}' using {search_api}")

        # Search the web with error handling
        try:
            if search_api == "tavily":
                search_results = tavily_search(state.search_query, fetch_full_page=configurable.fetch_full_page, max_results=1)
            elif search_api == "perplexity":
                search_results = perplexity_search(state.search_query, state.research_loop_count)
            elif search_api == "duckduckgo":
                search_results = duckduckgo_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
            elif search_api == "searxng":
                search_results = searxng_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
            else:
                raise ValueError(f"Unsupported search API: {configurable.search_api}")
            
            # Check if search_results is empty or None
            if not search_results:
                raise ValueError(f"No search results found for query: {state.search_query}")
                
        except Exception as e:
            # If search fails, create a fallback result
            print(f"Search error: {str(e)}")
            search_results = [{
                'title': f'Fallback information about {state.research_topic}',
                'content': f"No specific information found for '{state.search_query}'. "
                          f"This article will provide general information about {state.research_topic} "
                          f"based on recent trends and developments in 2025.",
                'url': 'https://example.com/fallback'
            }]

        # Format search results with error handling
        try:
            search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
        except Exception as format_err:
            # If formatting fails, create a simple fallback format
            print(f"Formatting error: {str(format_err)}")
            search_str = "\n\n".join([f"SOURCE: {r.get('title', 'Unknown Source')}\n{r.get('content', 'No content available')}" for r in search_results])

        # Check if search_str is empty and provide fallback
        if not search_str or len(search_str.strip()) < 50:
            search_str = f"Limited information found for '{state.search_query}'. "\
                        f"This topic ({state.research_topic}) may be very specialized or emerging. "\
                        f"The article will provide an overview based on available knowledge in 2025."

        # Print success message
        print(f"Successfully gathered {len(search_results)} search results")

        # Return the results
        return {
            "sources_gathered": [format_sources(search_results)], 
            "research_loop_count": state.research_loop_count + 1, 
            "web_research_results": [search_str]
        }

    except Exception as e:
        # Catch-all error handler
        print(f"Critical error in web_research: {str(e)}")
        
        # Create fallback research content
        fallback_content = f"Error encountered during web research: {str(e)}\n\n"\
                          f"This article about {state.research_topic} will provide general information "\
                          f"based on available knowledge without specific web research results."
        
        # Return fallback state update
        return {
            "sources_gathered": state.sources_gathered if hasattr(state, "sources_gathered") else 0, 
            "research_loop_count": state.research_loop_count + 1, 
            "web_research_results": [fallback_content]
        }

def article_writer(state: SummaryState, config: RunnableConfig):
    """LangGraph node that creates or enhances an SEO-optimized article.
    
    Uses an LLM to create or update an article based on the newest web research 
    results, integrating them with any existing content and optimizing for SEO.
    
    Args:
        state: Current graph state containing research topic, article content,
              web research results, and SEO keywords
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including article_content and seo_keywords
    """

    # Existing article content
    existing_article = state.article_content if hasattr(state, "article_content") else ""

    # Most recent web research with safeguards
    if hasattr(state, "web_research_results") and state.web_research_results:
        most_recent_web_research = state.web_research_results[-1]
    else:
        # Fallback if no research results exist
        print("Warning: No web research results found in state")
        most_recent_web_research = f"No research results available for topic: {state.research_topic}"
    
    # Get SEO keywords - either from state or from generate_query result
    seo_keywords = state.seo_keywords if state.seo_keywords else []
    
    # Format SEO keywords for the prompt
    seo_keywords_str = ", ".join(seo_keywords) if seo_keywords else "No specific keywords identified yet"

    # Enhanced HTML-specific instructions
    html_instructions = """
    HTML FORMATTING REQUIREMENTS:
    1. Use proper semantic HTML throughout the article
    2. Each main section should use <h2> tags with id attributes (<h2 id="section-name">Section Title</h2>)
    3. Subsections should use <h3> tags with id attributes
    4. Paragraphs should be wrapped in <p> tags
    5. Lists should use <ul> and <li> or <ol> and <li> tags
    6. Tables should use proper <table>, <tr>, <th>, and <td> elements
    7. Begin with a table of contents using nested lists with anchor links to each section
    8. DO NOT include <html>, <head>, <body>, <article>, <section>, or any document structure tags
    9. DO NOT include any CSS or JavaScript
    10. DO NOT write the title in an <h1> tag - that will be added later
    11. Start directly with the table of contents and then the first section
    """

    # Build the human message based on whether we're creating or enhancing an article
    if existing_article:
        # For existing articles, focus on ENHANCING rather than rewriting
        human_message_content = (
            f"ENHANCE THIS ARTICLE ABOUT: '{state.research_topic}'\n\n"
            f"<Article Topic> \n {state.research_topic} \n </Article Topic>\n\n"
            f"<SEO Keywords> \n {seo_keywords_str} \n </SEO Keywords>\n\n"
            f"<Existing Article> \n {existing_article} \n </Existing Article>\n\n"
            f"<New Research> \n {most_recent_web_research} \n </New Research>\n\n"
            f"ENHANCEMENT INSTRUCTIONS:\n"
            f"1. PRESERVE all existing content in the article\n"
            f"2. IDENTIFY where the new research information fits best in the existing content\n"
            f"3. INTEGRATE the new information into existing sections where relevant\n"
            f"4. ADD new sections only for completely new topics not covered in existing content\n"
            f"5. UPDATE the table of contents to reflect any new sections\n"
            f"6. MAINTAIN the existing HTML structure and formatting style\n\n"
            f"{html_instructions}\n\n"
            f"IMPORTANT: Your goal is to ENHANCE, not rewrite. Preserve all existing content and only add new information from the research materials.\n"
            f"NEVER remove or replace existing content - only add to it.\n"
            f"DO NOT start from scratch or rewrite the whole article - just enhance it with new information."
        )
    else:
        # For new articles, create comprehensive initial content
        human_message_content = (
            f"WRITE A NEW ARTICLE ABOUT: '{state.research_topic}'\n\n"
            f"<Article Topic> \n {state.research_topic} \n </Article Topic>\n\n"
            f"<SEO Keywords> \n {seo_keywords_str} \n </SEO Keywords>\n\n"
            f"<Research Materials> \n {most_recent_web_research} \n </Research Materials>\n\n"
            f"ARTICLE STRUCTURE REQUIREMENTS:\n"
            f"1. Begin with a table of contents using nested lists with anchor links to each section\n"
            f"2. Write a substantial, in-depth article (1500-2000+ words)\n"
            f"3. Organize into 5-7 well-structured sections with descriptive headers\n"
            f"4. Each section should have at least 3-4 paragraphs of detailed content\n"
            f"5. Use proper HTML anchors for all section headers (<h2 id=\"section-id\">Section Title</h2>)\n"
            f"6. Include comparison tables, lists, and examples where relevant\n\n"
            f"{html_instructions}\n\n"
            f"IMPORTANT: Your article must focus ONLY on '{state.research_topic}' using ONLY the provided research materials.\n"
            f"DO NOT include any document structure HTML tags like <html>, <head>, <body>, etc. - only content HTML."
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0
        )
    else:  # Default to Ollama
        llm = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0
        )
    
    try:
        # Make the LLM call with proper error handling
        result = llm.invoke(
            [SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content)]
        )

        # Strip thinking tokens if configured
        article_content = result.content
        if configurable.strip_thinking_tokens:
            article_content = strip_thinking_tokens(article_content)
            
        # Check if we got an empty or very short result (likely an error)
        if not article_content or len(article_content.strip()) < 50:
            # Create a fallback message as content
            article_content = f"""
            <h2 id="introduction">Introduction to {state.research_topic}</h2>
            <p>This article will explore {state.research_topic} in detail, based on the latest research and developments.</p>
            <p>Unfortunately, our AI writer encountered an issue generating the complete article content. 
            Please try running the process again or refining your research topic.</p>
            
            <h2 id="research-summary">Research Summary</h2>
            <p>Here's a summary of the research that was gathered:</p>
            <p>{most_recent_web_research[:500]}...</p>
            """
        
        print(f"Article writer generated content of length: {len(article_content)}")
        return {"article_content": article_content}
        
    except Exception as e:
        # In case of any exception, provide a fallback article
        error_message = str(e)
        print(f"Error in article_writer: {error_message}")
        
        fallback_content = f"""
        <h2 id="introduction">Introduction to {state.research_topic}</h2>
        <p>This article aims to explore {state.research_topic} in detail. However, we encountered a technical issue while generating content.</p>
        <p>Error details: {error_message}</p>
        
        <h2 id="research-summary">Research Summary</h2>
        <p>Here's a summary of the research that was gathered:</p>
        <p>{most_recent_web_research[:300]}...</p>
        """
        
        return {"article_content": fallback_content}

def seo_analysis(state: SummaryState, config: RunnableConfig):
    """LangGraph node that analyzes article content for SEO optimization opportunities.
    
    Analyzes the current article to identify content gaps, SEO opportunities, and generates
    follow-up queries to enhance article quality and search ranking. Uses structured output 
    to extract SEO recommendations and follow-up queries in JSON format.
    
    Args:
        state: Current graph state containing the article content and research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query and seo_keywords
    """

    # Get previous queries to avoid repetition
    previous_queries = []
    if hasattr(state, 'search_query') and state.search_query:
        previous_queries.append(state.search_query)
    
    # Format previous queries
    prev_queries_str = ", ".join(previous_queries) if previous_queries else "No previous queries"

    # Generate a more focused instruction that emphasizes identifying gaps and missing information
    enhanced_instruction = f"""
    <GOAL>
    Analyze an article about "{state.research_topic}" to identify SPECIFIC content gaps and missing information
    that should be researched further to enhance the article. Your primary job is to find what's MISSING.
    </GOAL>

    <RULES>
    1. Focus EXCLUSIVELY on identifying MISSING sections, data points, examples, or perspectives
    2. Target your analysis on what information would COMPLEMENT the existing content, not replace it
    3. All keywords MUST be directly relevant to the exact topic "{state.research_topic}"
    4. Create follow-up queries designed to fill ONLY the identified information gaps
    5. Be extremely specific about what information is needed - avoid general queries
    6. If a major section is missing, specifically identify what that section should cover
    7. Prioritize depth over breadth - identify where existing sections lack sufficient detail
    </RULES>
    """

    # Generate SEO analysis
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    else: # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=enhanced_instruction + reflection_instructions.format(
            research_topic=state.research_topic,
            previous_queries=prev_queries_str,
            current_summary=state.article_content or "No article content yet."
        )),
        HumanMessage(content=f"Analyze this article about '{state.research_topic}' and identify SPECIFIC MISSING INFORMATION and content gaps that need to be researched further to make the article more comprehensive:")]
    )
    
    # Process the result
    try:
        # Try to parse as JSON first
        seo_analysis = json.loads(result.content)
        
        # Get the follow-up query
        query = seo_analysis.get('follow_up_query')
        
        # Get SEO keywords and ensure they're unique and relevant
        new_keywords = seo_analysis.get('target_keywords', [])
        
        # Filter keywords to ensure they're related to the research topic
        # Split the research topic into words to check for relevance
        topic_words = set(word.lower() for word in state.research_topic.split() if len(word) > 3)
        filtered_keywords = []
        
        for keyword in new_keywords:
            # Check if the keyword has at least one word from the research topic or is directly relevant
            # This is a simple heuristic and could be improved
            keyword_words = set(word.lower() for word in keyword.split() if len(word) > 3)
            if keyword_words.intersection(topic_words) or state.research_topic.lower() in keyword.lower():
                filtered_keywords.append(keyword)
        
        # Extract section gaps and content opportunities if available
        section_gaps = seo_analysis.get('section_gaps', [])
        content_opportunities = seo_analysis.get('content_opportunities', [])
        
        # For logging/debugging - print what we found
        print(f"Found {len(new_keywords)} keywords, filtered to {len(filtered_keywords)}, {len(section_gaps)} section gaps, and {len(content_opportunities)} content opportunities")
        
        # Check if query is None or empty
        if not query:
            # Use a fallback query
            query = f"Tell me more about {state.research_topic}"
        elif state.research_topic.lower() not in query.lower():
            # Ensure the query contains the research topic
            query = f"{query} specifically about {state.research_topic}"
            
        # Add the new keywords to the state - properly deduplicate
        existing_keywords = state.seo_keywords if state.seo_keywords else []
        
        # Normalize and deduplicate keywords (case-insensitive)
        normalized_existing = [k.lower().strip() for k in existing_keywords]
        all_keywords = existing_keywords.copy()
        
        for keyword in filtered_keywords:
            # Only add if normalized version isn't already in list
            normalized = keyword.lower().strip()
            if normalized not in normalized_existing:
                all_keywords.append(keyword)
                normalized_existing.append(normalized)
            
        return {"search_query": query, "seo_keywords": all_keywords}
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails or the key is not found, use a fallback query
        return {"search_query": f"Tell me more about {state.research_topic}"}
        
# The finalize_article function has been removed as it was redundant
# The article_writer function now directly produces the final HTML content

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["web_research", END]:
    """LangGraph routing function that determines whether to continue research or end the process.
    
    Controls the research loop by deciding whether to continue gathering information
    or to end the process when the maximum number of research loops is reached.
    
    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting
        
    Returns:
        String literal indicating the next node to visit ("web_research" or END)
    """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return END

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("article_writer", article_writer)
builder.add_node("seo_analysis", seo_analysis)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "article_writer")
builder.add_edge("article_writer", "seo_analysis")
builder.add_conditional_edges("seo_analysis", route_research)

graph = builder.compile()