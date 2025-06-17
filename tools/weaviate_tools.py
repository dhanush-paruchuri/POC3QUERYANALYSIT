# tools/weaviate_tools.py - CORRECTED VERSION WITH BUSINESS CONTEXT ANALYZER

import os
import json
import logging
import weaviate
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from crewai.tools import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import new timeout configuration classes
try:
    from weaviate.classes.init import AdditionalConfig, Timeout
    HAS_NEW_TIMEOUT_CONFIG = True
except ImportError:
    HAS_NEW_TIMEOUT_CONFIG = False
    logger.warning("Using older Weaviate client version - timeout configuration will be simplified")

# Load environment variables
load_dotenv()

# --- Simple Weaviate Client (No Over-Engineering) ---
class WeaviateClientSingleton:
    _instance: Optional[weaviate.WeaviateClient] = None
    _connection_attempted = False

    @classmethod
    def get_instance(cls) -> Optional[weaviate.WeaviateClient]:
        if cls._instance is None and not cls._connection_attempted:
            cls._connection_attempted = True
            logger.info("🔌 Connecting to Weaviate...")
            try:
                # Try multiple connection methods with proper timeout configuration
                weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
                
                if "localhost" in weaviate_url or "127.0.0.1" in weaviate_url:
                    # Use connect_to_local for local connections
                    if HAS_NEW_TIMEOUT_CONFIG:
                        # New timeout configuration method
                        timeout_config = AdditionalConfig(
                            timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
                        )
                        cls._instance = weaviate.connect_to_local(
                            port=8080,
                            grpc_port=50051,
                            additional_config=timeout_config
                        )
                    else:
                        # Fallback for older versions - try without timeout first
                        try:
                            cls._instance = weaviate.connect_to_local(
                                port=8080,
                                grpc_port=50051
                            )
                        except Exception:
                            # If that fails, try even simpler connection
                            cls._instance = weaviate.connect_to_local()
                else:
                    # Use connect_to_custom for remote connections
                    host = weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
                    if HAS_NEW_TIMEOUT_CONFIG:
                        timeout_config = AdditionalConfig(
                            timeout=Timeout(init=30, query=60, insert=120)
                        )
                        cls._instance = weaviate.connect_to_custom(
                            http_host=host,
                            http_port=8080,
                            http_secure=False,
                            additional_config=timeout_config
                        )
                    else:
                        cls._instance = weaviate.connect_to_custom(
                            http_host=host,
                            http_port=8080,
                            http_secure=False
                        )
                
                if cls._instance and cls._instance.is_ready():
                    logger.info("✅ Weaviate connection successful")
                    try:
                        collections = cls._instance.collections.list_all()
                        collection_names = [col.name for col in collections]
                        logger.info(f"Available collections: {collection_names}")
                    except Exception as e:
                        logger.warning(f"Could not list collections: {e}")
                else:
                    logger.warning("❌ Weaviate not ready, will use fallback data")
                    if cls._instance:
                        cls._instance.close()
                    cls._instance = None
            except Exception as e:
                logger.warning(f"❌ Weaviate connection failed: {e}, will use fallback data")
                if cls._instance:
                    try:
                        cls._instance.close()
                    except:
                        pass
                cls._instance = None
        return cls._instance

    @classmethod
    def close(cls):
        if cls._instance:
            cls._instance.close()
            cls._instance = None

@tool("Business Context Analyzer")  
def business_context_analyzer(query: str) -> str:
    """
    STRATEGIC BUSINESS INTELLIGENCE EXTRACTION
    
    This tool extracts business context to MAXIMIZE Weaviate retrieval efficiency.
    The output is specifically designed to make the Hierarchical Schema Discovery
    tool run faster and more accurately by providing:
    
    1. Targeted search queries for each business entity
    2. Column priority rankings for faster discovery
    3. Relationship mapping hints for efficient joins
    4. Performance optimization strategies
    
    This is the first step in an intelligent retrieval pipeline.
    """
    weaviate_client = WeaviateClientSingleton.get_instance()
    logger.info(f"🧠 STRATEGIC Business Intelligence Extraction for: '{query}'")
    
    # PHASE 1: Business Intent Analysis
    business_intel = _extract_strategic_business_intelligence(query)
    
    # PHASE 2: Generate Weaviate Optimization Strategy
    weaviate_strategy = _generate_weaviate_optimization_strategy(business_intel)
    
    # PHASE 3: Enhanced with Weaviate BusinessContext (if available)
    if weaviate_client:
        try:
            enhanced_intel = _enhance_with_weaviate_business_context(query, business_intel, weaviate_client)
            business_intel.update(enhanced_intel)
            logger.info("✅ Enhanced with Weaviate BusinessContext data")
        except Exception as e:
            logger.warning(f"Could not enhance with Weaviate: {e}")
    
    # PHASE 4: Create Final Strategic Output
    strategic_output = {
        "query": query,
        "business_intelligence": business_intel,
        "weaviate_optimization_strategy": weaviate_strategy,
        "retrieval_instructions": {
            "priority_search_order": _create_priority_search_order(business_intel),
            "targeted_queries": _create_targeted_weaviate_queries(business_intel),
            "column_priority_ranking": _create_column_priority_ranking(business_intel),
            "relationship_discovery_hints": _create_relationship_hints(business_intel),
            "performance_optimizations": _create_performance_optimizations(business_intel)
        },
        "expected_schema_elements": {
            "primary_tables": _predict_primary_tables(business_intel),
            "critical_columns": _predict_critical_columns(business_intel),
            "likely_relationships": _predict_relationships(business_intel)
        }
    }
    
    logger.info("✅ Strategic Business Intelligence extraction complete")
    return json.dumps(strategic_output, indent=2)

def _extract_strategic_business_intelligence(query: str) -> Dict[str, Any]:
    """Extract strategic business intelligence for Weaviate optimization."""
    query_lower = query.lower()
    
    # Intent Classification (affects search strategy)
    intent_map = {
        "contact_lookup": any(word in query_lower for word in ['email', 'phone', 'contact', 'address']),
        "customer_analysis": any(word in query_lower for word in ['customer', 'client', 'who']),
        "operational_metrics": any(word in query_lower for word in ['count', 'how many', 'total', 'number']),
        "ranking_analysis": any(word in query_lower for word in ['most', 'top', 'best', 'highest']),
        "temporal_analysis": any(word in query_lower for word in ['when', 'date', '2024', '2023', 'year']),
        "financial_analysis": any(word in query_lower for word in ['cost', 'price', 'payment', 'billing'])
    }
    
    primary_intent = max(intent_map.items(), key=lambda x: x[1])[0] if any(intent_map.values()) else "general_lookup"
    
    # Entity Extraction (determines table priorities)
    entities = {
        "customer": any(word in query_lower for word in ['customer', 'client', 'user', 'ashraf', 'john', 'name']),
        "moves": any(word in query_lower for word in ['move', 'booking', 'order', 'service', 'relocation']),
        "contact": any(word in query_lower for word in ['email', 'phone', 'contact', 'address']),
        "temporal": any(word in query_lower for word in ['date', 'time', 'when', '2024', '2023']),
        "financial": any(word in query_lower for word in ['cost', 'price', 'payment', 'billing'])
    }
    
    active_entities = [entity for entity, present in entities.items() if present]
    
    # Business Complexity Assessment
    complexity = "simple" if len(active_entities) <= 1 else "moderate" if len(active_entities) <= 2 else "complex"
    
    return {
        "primary_intent": primary_intent,
        "active_entities": active_entities,
        "complexity_level": complexity,
        "domain": "moving_services",
        "query_patterns": _identify_query_patterns(query_lower),
        "urgency_level": "high" if primary_intent == "contact_lookup" else "medium",
        "data_scope": "single_record" if "ashraf" in query_lower else "aggregate"
    }

def _generate_weaviate_optimization_strategy(business_intel: Dict[str, Any]) -> Dict[str, Any]:
    """Generate optimization strategy for Weaviate queries."""
    primary_intent = business_intel["primary_intent"]
    active_entities = business_intel["active_entities"]
    complexity = business_intel["complexity_level"]
    
    strategy = {
        "search_approach": "targeted_sequential",  # vs broad_parallel
        "max_queries_per_collection": 3 if complexity == "simple" else 5,
        "prioritize_precision": primary_intent == "contact_lookup",
        "enable_relationship_discovery": len(active_entities) > 1,
        "use_semantic_boosting": True
    }
    
    if primary_intent == "contact_lookup":
        strategy.update({
            "search_approach": "laser_focused",
            "max_queries_per_collection": 2,  # Minimize queries for speed
            "prioritize_precision": True,
            "focus_entities": ["customer", "contact"]
        })
    
    return strategy

def _create_priority_search_order(business_intel: Dict[str, Any]) -> List[str]:
    """Create priority order for searching Weaviate collections."""
    primary_intent = business_intel["primary_intent"]
    active_entities = business_intel["active_entities"]
    
    if primary_intent == "contact_lookup":
        return ["DatasetMetadata->customer_tables", "Column->contact_columns", "BusinessContext->contact_terms"]
    
    if "customer" in active_entities and "moves" in active_entities:
        return ["DatasetMetadata->customer_moves", "Column->customer_move_columns", "DataRelationship->customer_move_joins"]
    
    if "customer" in active_entities:
        return ["DatasetMetadata->customer_tables", "Column->customer_columns"]
    
    return ["DatasetMetadata->general", "Column->general"]

def _create_targeted_weaviate_queries(business_intel: Dict[str, Any]) -> Dict[str, List[str]]:
    """Create highly targeted Weaviate queries based on business intelligence."""
    primary_intent = business_intel["primary_intent"]
    active_entities = business_intel["active_entities"]
    
    queries = {
        "dataset_queries": [],
        "column_queries": [],
        "relationship_queries": []
    }
    
    if primary_intent == "contact_lookup":
        queries["dataset_queries"] = [
            "customer master data contact information CRM",
            "customer table email phone address contact"
        ]
        queries["column_queries"] = [
            "email address email_address customer_email contact_email",
            "phone number phone_number customer_phone contact_phone",
            "first_name last_name customer_name name identification"
        ]
    
    elif primary_intent == "customer_analysis":
        queries["dataset_queries"] = [
            "customer master data demographics CRM client",
            "customer table customer_id first_name last_name"
        ]
        queries["column_queries"] = [
            "customer_id customer_name first_name last_name",
            "customer identification customer demographics"
        ]
    
    elif primary_intent == "operational_metrics":
        queries["dataset_queries"] = [
            "moves bookings operations service_orders transactions",
            "operational data move_table booking_table"
        ]
        queries["column_queries"] = [
            "move_id booking_id order_id transaction_id",
            "count total quantity number volume"
        ]
    
    # Add relationship queries if multiple entities
    if len(active_entities) > 1:
        if "customer" in active_entities and "moves" in active_entities:
            queries["relationship_queries"] = [
                "customer moves relationship customer_id foreign_key",
                "customer booking join customer_moves relationship"
            ]
    
    return queries

def _create_column_priority_ranking(business_intel: Dict[str, Any]) -> Dict[str, int]:
    """Create priority ranking for columns to optimize search order."""
    primary_intent = business_intel["primary_intent"]
    
    if primary_intent == "contact_lookup":
        return {
            "email": 10, "email_address": 10, "customer_email": 9,
            "phone": 9, "phone_number": 9, "contact_phone": 8,
            "first_name": 8, "last_name": 8, "customer_name": 7,
            "id": 6, "customer_id": 6
        }
    
    elif primary_intent == "customer_analysis":
        return {
            "customer_id": 10, "first_name": 9, "last_name": 9,
            "customer_name": 8, "id": 7, "email": 6
        }
    
    elif primary_intent == "operational_metrics":
        return {
            "id": 10, "move_id": 10, "customer_id": 9,
            "status": 8, "count": 7, "date": 7
        }
    
    return {"id": 5, "name": 4, "date": 3}

def _create_relationship_hints(business_intel: Dict[str, Any]) -> List[str]:
    """Create specific hints for relationship discovery."""
    active_entities = business_intel["active_entities"]
    hints = []
    
    if "customer" in active_entities and "moves" in active_entities:
        hints.extend([
            "customer.id -> moves.customer_id (PRIMARY relationship)",
            "customer_id foreign_key relationship highest_priority",
            "customer moves join customer_table move_table"
        ])
    
    if "customer" in active_entities and "contact" in active_entities:
        hints.extend([
            "customer contact information same_table relationship",
            "customer_email customer_phone embedded_in_customer_table"
        ])
    
    return hints

def _create_performance_optimizations(business_intel: Dict[str, Any]) -> List[str]:
    """Create performance optimization strategies."""
    complexity = business_intel["complexity_level"]
    data_scope = business_intel["data_scope"]
    
    optimizations = []
    
    if data_scope == "single_record":
        optimizations.extend([
            "LIMIT Weaviate queries to 5 results max for speed",
            "Use exact term matching where possible",
            "Prioritize precision over recall for single record lookup"
        ])
    
    if complexity == "simple":
        optimizations.extend([
            "Use sequential search pattern - table first, then columns",
            "Minimize collection queries - focus on most relevant"
        ])
    
    optimizations.extend([
        "Use semantic distance threshold of 0.7 for relevance filtering",
        "Cache results for repeated entity combinations",
        "Batch similar queries together"
    ])
    
    return optimizations

def _predict_primary_tables(business_intel: Dict[str, Any]) -> List[str]:
    """Predict primary tables needed based on business intelligence."""
    active_entities = business_intel["active_entities"]
    primary_intent = business_intel["primary_intent"]
    
    tables = []
    
    if "customer" in active_entities or primary_intent == "contact_lookup":
        tables.append("customer")
    
    if "moves" in active_entities:
        tables.append("moves")
    
    if "financial" in active_entities:
        tables.append("billing")
    
    return tables

def _predict_critical_columns(business_intel: Dict[str, Any]) -> List[str]:
    """Predict critical columns based on business intelligence."""
    primary_intent = business_intel["primary_intent"]
    
    if primary_intent == "contact_lookup":
        return ["email", "phone", "first_name", "last_name", "customer_id"]
    
    elif primary_intent == "customer_analysis":
        return ["customer_id", "first_name", "last_name", "customer_name"]
    
    elif primary_intent == "operational_metrics":
        return ["id", "customer_id", "status", "date", "count"]
    
    return ["id", "name", "date"]

def _predict_relationships(business_intel: Dict[str, Any]) -> List[Dict[str, str]]:
    """Predict likely relationships based on business intelligence."""
    active_entities = business_intel["active_entities"]
    relationships = []
    
    if "customer" in active_entities and "moves" in active_entities:
        relationships.append({
            "from": "customer",
            "to": "moves", 
            "via": "customer_id",
            "type": "one_to_many"
        })
    
    return relationships

def _identify_query_patterns(query_lower: str) -> List[str]:
    """Identify specific query patterns for optimization."""
    patterns = []
    
    if any(name in query_lower for name in ['ashraf', 'john', 'smith', 'jones']):
        patterns.append("named_entity_lookup")
    
    if "email" in query_lower:
        patterns.append("contact_retrieval")
    
    if any(word in query_lower for word in ['count', 'how many', 'total']):
        patterns.append("aggregation_query")
    
    return patterns

def _enhance_with_weaviate_business_context(query: str, business_intel: Dict[str, Any], client) -> Dict[str, Any]:
    """Enhance business intelligence with actual Weaviate BusinessContext data."""
    try:
        business_collection = client.collections.get("BusinessContext")
        
        # Use targeted search based on business intelligence
        search_terms = []
        if business_intel["primary_intent"] == "contact_lookup":
            search_terms.extend(["contact information", "customer communication", "email phone"])
        
        search_terms.append(query)  # Always include original
        
        enhanced_terms = []
        for search_term in search_terms:
            response = business_collection.query.near_text(
                query=search_term,
                limit=2,  # Limit for speed
                return_properties=["term", "definition", "context"],
                return_metadata=['distance']
            )
            
            for obj in response.objects:
                if obj.metadata.distance < 0.3:  # High relevance only
                    enhanced_terms.append({
                        "term": obj.properties.get('term', ''),
                        "definition": obj.properties.get('definition', ''),
                        "relevance": 1 - obj.metadata.distance
                    })
        
        return {"enhanced_business_terms": enhanced_terms}
        
    except Exception as e:
        logger.warning(f"BusinessContext enhancement failed: {e}")
        return {}

# Clean up on module exit
import atexit
atexit.register(WeaviateClientSingleton.close)

@tool("Hierarchical Schema Discovery")
def hierarchical_schema_discovery(query: str) -> str:
    """
    INTELLIGENT WEAVIATE RETRIEVAL ENGINE
    
    This tool uses strategic business intelligence from the Business Context Analyzer
    to make HIGHLY OPTIMIZED Weaviate queries for maximum accuracy and speed.
    
    Key optimizations:
    1. Uses targeted queries based on business intent
    2. Follows priority search order for speed  
    3. Applies column priority ranking for efficiency
    4. Uses relationship hints for smart join discovery
    5. Implements performance optimizations throughout
    
    This is the second step in the intelligent retrieval pipeline.
    """
    weaviate_client = WeaviateClientSingleton.get_instance()
    logger.info(f"🚀 INTELLIGENT Weaviate Retrieval Engine for: '{query}'")

    if not weaviate_client:
        logger.warning("🔄 Weaviate not available, using optimized fallback")
        return _get_optimized_fallback_schema(query)

    # STEP 1: Extract Strategic Business Intelligence from Query
    business_strategy = _extract_business_strategy_from_query(query)
    logger.info(f"📋 Business Strategy: {business_strategy['approach']} | Entities: {business_strategy['priority_entities']}")

    results = {
        "query": query,
        "business_strategy_applied": business_strategy,
        "datasets": [],
        "columns_by_dataset": {},
        "relationships": [],
        "optimization_metrics": {
            "queries_executed": 0,
            "results_filtered": 0,
            "efficiency_score": 0
        }
    }

    try:
        # STEP 2: Intelligent Dataset Discovery (Optimized)
        logger.info("1️⃣ TARGETED Dataset Discovery...")
        
        datasets_found = _execute_optimized_dataset_search(
            weaviate_client, 
            business_strategy, 
            results["optimization_metrics"]
        )
        
        results["datasets"] = datasets_found
        relevant_table_names = [d["table_name"] for d in datasets_found]
        
        if not relevant_table_names:
            logger.warning("No relevant datasets found")
            return json.dumps(results, indent=2)

        # STEP 3: Intelligent Column Discovery (Priority-Based)
        logger.info("2️⃣ PRIORITY-BASED Column Discovery...")
        
        columns_found = _execute_optimized_column_search(
            weaviate_client,
            business_strategy,
            relevant_table_names,
            results["datasets"],
            results["optimization_metrics"]
        )
        
        results["columns_by_dataset"] = columns_found

        # STEP 4: Smart Relationship Discovery (Hint-Driven)
        logger.info("3️⃣ HINT-DRIVEN Relationship Discovery...")
        
        if len(relevant_table_names) > 1 and business_strategy["enable_relationships"]:
            relationships_found = _execute_optimized_relationship_search(
                weaviate_client,
                business_strategy,
                relevant_table_names,
                results["optimization_metrics"]
            )
            results["relationships"] = relationships_found
        
        # STEP 5: Calculate Efficiency Metrics
        results["optimization_metrics"]["efficiency_score"] = _calculate_efficiency_score(
            results["optimization_metrics"],
            len(results["datasets"]),
            sum(len(cols) for cols in results["columns_by_dataset"].values()),
            len(results["relationships"])
        )
        
        logger.info(f"✅ Optimized retrieval complete - Efficiency: {results['optimization_metrics']['efficiency_score']:.2f}")
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Optimized search error: {e}", exc_info=True)
        return _get_optimized_fallback_schema(query)

def _extract_business_strategy_from_query(query: str) -> Dict[str, Any]:
    """Extract or parse business strategy from the query context."""
    try:
        # Try to extract business intelligence from query if it contains JSON
        if "weaviate_optimization_strategy" in query:
            import re
            json_match = re.search(r'\{.*"weaviate_optimization_strategy".*\}', query, re.DOTALL)
            if json_match:
                context_data = json.loads(json_match.group())
                strategy = context_data.get("weaviate_optimization_strategy", {})
                return {
                    "approach": strategy.get("search_approach", "targeted_sequential"),
                    "max_queries": strategy.get("max_queries_per_collection", 3),
                    "prioritize_precision": strategy.get("prioritize_precision", False),
                    "enable_relationships": strategy.get("enable_relationship_discovery", True),
                    "priority_entities": context_data.get("business_intelligence", {}).get("active_entities", []),
                    "primary_intent": context_data.get("business_intelligence", {}).get("primary_intent", "general"),
                    "targeted_queries": context_data.get("retrieval_instructions", {}).get("targeted_queries", {}),
                    "column_priorities": context_data.get("retrieval_instructions", {}).get("column_priority_ranking", {}),
                    "performance_opts": context_data.get("retrieval_instructions", {}).get("performance_optimizations", [])
                }
    except:
        pass
    
    # Fallback: Create strategy from raw query analysis
    query_lower = query.lower()
    
    # Determine approach
    approach = "laser_focused" if any(name in query_lower for name in ['ashraf', 'john', 'smith']) else "targeted_sequential"
    
    # Identify entities
    priority_entities = []
    if any(word in query_lower for word in ['customer', 'client', 'ashraf', 'name']):
        priority_entities.append("customer")
    if any(word in query_lower for word in ['email', 'phone', 'contact', 'address']):
        priority_entities.append("contact")
    if any(word in query_lower for word in ['move', 'booking', 'order']):
        priority_entities.append("moves")
    
    # Determine intent
    primary_intent = "contact_lookup" if "email" in query_lower else "customer_analysis" if priority_entities else "general"
    
    return {
        "approach": approach,
        "max_queries": 2 if approach == "laser_focused" else 3,
        "prioritize_precision": primary_intent == "contact_lookup",
        "enable_relationships": len(priority_entities) > 1,
        "priority_entities": priority_entities,
        "primary_intent": primary_intent,
        "targeted_queries": _generate_fallback_targeted_queries(primary_intent, priority_entities),
        "column_priorities": _generate_fallback_column_priorities(primary_intent),
        "performance_opts": ["minimize_queries", "prioritize_precision"] if approach == "laser_focused" else ["standard_optimization"]
    }

def _execute_optimized_dataset_search(weaviate_client, business_strategy: Dict[str, Any], metrics: Dict[str, int]) -> List[Dict[str, Any]]:
    """Execute optimized dataset search using business strategy."""
    dataset_collection = weaviate_client.collections.get("DatasetMetadata")
    datasets_found = []
    dataset_relevance_scores = {}
    
    # Get targeted queries from business strategy
    targeted_queries = business_strategy.get("targeted_queries", {}).get("dataset_queries", [])
    if not targeted_queries:
        targeted_queries = [business_strategy.get("query", "customer data")]
    
    max_queries = business_strategy.get("max_queries", 3)
    prioritize_precision = business_strategy.get("prioritize_precision", False)
    
    # Execute targeted searches
    for i, search_query in enumerate(targeted_queries[:max_queries]):
        logger.info(f"   🔍 Dataset search {i+1}: '{search_query}'")
        
        distance_threshold = 0.3 if prioritize_precision else 0.6
        
        try:
            response = dataset_collection.query.near_text(
                query=search_query,
                limit=2 if prioritize_precision else 3,
                return_metadata=['distance'],
                return_properties=["tableName", "athenaTableName", "description", "recordCount"]
            )
            
            metrics["queries_executed"] += 1
            
            for obj in response.objects:
                props = obj.properties
                table_name = props.get('tableName', '')
                distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0
                relevance = 1 - distance
                
                if table_name and relevance > (1 - distance_threshold):
                    if table_name not in dataset_relevance_scores:
                        dataset_relevance_scores[table_name] = relevance
                        
                        athena_table = props.get('athenaTableName', table_name)
                        if "." not in athena_table:
                            athena_table = f"{athena_table}"
                        
                        datasets_found.append({
                            "table_name": table_name,
                            "athena_table_name": athena_table,
                            "relevance": relevance,
                            "description": props.get('description', ''),
                            "record_count": props.get('recordCount', 0),
                            "search_query_used": search_query,
                            "business_relevance": _explain_dataset_business_relevance(table_name, business_strategy)
                        })
                        
                        logger.info(f"      ✅ {table_name} (relevance: {relevance:.3f})")
                else:
                    metrics["results_filtered"] += 1
                    
        except Exception as e:
            logger.warning(f"Dataset search failed for '{search_query}': {e}")
    
    # Sort by relevance
    datasets_found.sort(key=lambda x: x["relevance"], reverse=True)
    
    return datasets_found

def _execute_optimized_column_search(weaviate_client, business_strategy: Dict[str, Any], 
                                   relevant_table_names: List[str], datasets: List[Dict[str, Any]], 
                                   metrics: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
    """Execute optimized column search using priority ranking."""
    column_collection = weaviate_client.collections.get("Column")
    columns_by_dataset = {}
    
    # Get column priorities and targeted queries
    column_priorities = business_strategy.get("column_priorities", {})
    targeted_queries = business_strategy.get("targeted_queries", {}).get("column_queries", [])
    
    if not targeted_queries:
        primary_intent = business_strategy.get("primary_intent", "general")
        if primary_intent == "contact_lookup":
            targeted_queries = ["email address contact phone", "first_name last_name customer_name"]
        else:
            targeted_queries = ["customer_id first_name last_name", "id primary_key"]
    
    max_queries = business_strategy.get("max_queries", 3)
    
    # Execute targeted column searches
    for search_query in targeted_queries[:max_queries]:
        logger.info(f"   🔍 Column search: '{search_query}'")
        
        try:
            response = column_collection.query.near_text(
                query=search_query,
                limit=30,  # Get more columns to filter by table relevance
                return_metadata=['distance'],
                return_properties=["columnName", "dataType", "description", "datasetTableName", 
                                 "isPrimaryKey", "isForeignKey", "sampleValues"]
            )
            
            metrics["queries_executed"] += 1
            
            for col_obj in response.objects:
                col_props = col_obj.properties
                col_table = col_props.get('datasetTableName', '')
                col_name = col_props.get('columnName', '')
                
                # Only process columns from relevant tables
                if col_table in relevant_table_names:
                    col_distance = col_obj.metadata.distance if hasattr(col_obj.metadata, 'distance') else 1.0
                    col_relevance = 1 - col_distance
                    
                    # Apply priority boost if column is in priority list
                    priority_boost = column_priorities.get(col_name.lower(), 0) * 0.1
                    adjusted_relevance = min(1.0, col_relevance + priority_boost)
                    
                    # Filter by adjusted relevance
                    if adjusted_relevance > 0.4:  # Reasonable threshold
                        # Find matching dataset
                        matching_dataset = next(
                            (d for d in datasets if d["table_name"] == col_table), None
                        )
                        
                        if matching_dataset:
                            athena_table = matching_dataset["athena_table_name"]
                            
                            if athena_table not in columns_by_dataset:
                                columns_by_dataset[athena_table] = []
                            
                            # Check for duplicates
                            existing_cols = [c["column_name"] for c in columns_by_dataset[athena_table]]
                            if col_name not in existing_cols:
                                columns_by_dataset[athena_table].append({
                                    "column_name": col_name,
                                    "data_type": col_props.get('dataType'),
                                    "description": col_props.get('description', ''),
                                    "relevance": adjusted_relevance,
                                    "original_relevance": col_relevance,
                                    "priority_boost": priority_boost,
                                    "is_primary_key": col_props.get('isPrimaryKey', False),
                                    "is_foreign_key": col_props.get('isForeignKey', False),
                                    "sample_values": col_props.get('sampleValues', []),
                                    "business_relevance": _explain_column_business_relevance(col_name, business_strategy)
                                })
                    else:
                        metrics["results_filtered"] += 1
                        
        except Exception as e:
            logger.warning(f"Column search failed for '{search_query}': {e}")
    
    # Sort columns by adjusted relevance within each dataset  
    for table_name in columns_by_dataset:
        columns_by_dataset[table_name].sort(key=lambda x: x['relevance'], reverse=True)
        
        # Show top columns
        logger.info(f"   📊 {table_name}:")
        for col in columns_by_dataset[table_name][:5]:
            boost_indicator = f" (+{col['priority_boost']:.1f})" if col['priority_boost'] > 0 else ""
            logger.info(f"      • {col['column_name']} ({col['data_type']}) - {col['relevance']:.3f}{boost_indicator}")
    
    return columns_by_dataset

def _execute_optimized_relationship_search(weaviate_client, business_strategy: Dict[str, Any],
                                         relevant_table_names: List[str], metrics: Dict[str, int]) -> List[Dict[str, Any]]:
    """Execute optimized relationship search using business hints."""
    rel_collection = weaviate_client.collections.get("DataRelationship")
    relationships_found = []
    
    # Get relationship queries from strategy
    targeted_queries = business_strategy.get("targeted_queries", {}).get("relationship_queries", [])
    
    if not targeted_queries:
        # Generate based on entities
        priority_entities = business_strategy.get("priority_entities", [])
        if "customer" in priority_entities and any(e in priority_entities for e in ["moves", "contact"]):
            targeted_queries = ["customer moves relationship customer_id", "customer join foreign_key"]
    
    if not targeted_queries:
        # Fallback to general relationship search
        targeted_queries = [f"{relevant_table_names[0]} {relevant_table_names[1]} relationship"]
    
    for search_query in targeted_queries[:2]:  # Limit relationship searches
        logger.info(f"   🔍 Relationship search: '{search_query}'")
        
        try:
            response = rel_collection.query.near_text(
                query=search_query,
                limit=5,
                return_properties=["fromTableName", "fromColumn", "toTableName", 
                                 "toColumn", "suggestedJoinType", "relationshipStrength"]
            )
            
            metrics["queries_executed"] += 1
            
            for obj in response.objects:
                props = obj.properties
                from_table = props.get('fromTableName')
                to_table = props.get('toTableName')
                
                if from_table in relevant_table_names and to_table in relevant_table_names:
                    rel_key = f"{from_table}.{props.get('fromColumn')}->{to_table}.{props.get('toColumn')}"
                    existing_rels = [f"{r['from_table']}.{r['from_column']}->{r['to_table']}.{r['to_column']}" 
                                   for r in relationships_found]
                    
                    if rel_key not in existing_rels:
                        relationships_found.append({
                            "from_table": from_table,
                            "from_column": props.get('fromColumn'),
                            "to_table": to_table,
                            "to_column": props.get('toColumn'),
                            "join_type": props.get('suggestedJoinType', 'INNER'),
                            "relationship_strength": props.get('relationshipStrength', 'medium'),
                            "business_purpose": _explain_relationship_business_purpose(from_table, to_table, business_strategy)
                        })
                        
                        logger.info(f"      ✅ {from_table}.{props.get('fromColumn')} → {to_table}.{props.get('toColumn')}")
                        
        except Exception as e:
            logger.warning(f"Relationship search failed for '{search_query}': {e}")
    
    return relationships_found

def _calculate_efficiency_score(metrics: Dict[str, int], datasets_found: int, columns_found: int, relationships_found: int) -> float:
    """Calculate efficiency score based on results vs queries executed."""
    queries_executed = metrics.get("queries_executed", 1)
    results_filtered = metrics.get("results_filtered", 0)
    
    total_results = datasets_found + columns_found + relationships_found
    
    if queries_executed == 0:
        return 0.0
    
    # Base efficiency: results per query
    base_efficiency = total_results / queries_executed
    
    # Bonus for high-quality filtering (filtering out irrelevant results)
    filter_bonus = min(0.3, results_filtered / (results_filtered + total_results)) if (results_filtered + total_results) > 0 else 0
    
    # Normalize to 0-1 scale
    efficiency_score = min(1.0, (base_efficiency / 10) + filter_bonus)
    
    return efficiency_score

def _generate_fallback_targeted_queries(primary_intent: str, priority_entities: List[str]) -> Dict[str, List[str]]:
    """Generate fallback targeted queries when business strategy is not available."""
    queries = {"dataset_queries": [], "column_queries": [], "relationship_queries": []}
    
    if primary_intent == "contact_lookup":
        queries["dataset_queries"] = ["customer master data contact", "customer table CRM"]
        queries["column_queries"] = ["email address contact", "first_name last_name phone"]
    
    elif "customer" in priority_entities:
        queries["dataset_queries"] = ["customer master data", "customer table demographics"]
        queries["column_queries"] = ["customer_id first_name last_name", "customer identification"]
    
    if len(priority_entities) > 1:
        queries["relationship_queries"] = [f"{priority_entities[0]} {priority_entities[1]} relationship"]
    
    return queries

def _generate_fallback_column_priorities(primary_intent: str) -> Dict[str, int]:
    """Generate fallback column priorities."""
    if primary_intent == "contact_lookup":
        return {"email": 10, "phone": 9, "first_name": 8, "last_name": 8}
    return {"id": 5, "customer_id": 7, "name": 6}

def _explain_dataset_business_relevance(table_name: str, business_strategy: Dict[str, Any]) -> str:
    """Explain why a dataset is relevant to the business strategy."""
    primary_intent = business_strategy.get("primary_intent", "")
    priority_entities = business_strategy.get("priority_entities", [])
    
    if primary_intent == "contact_lookup" and "customer" in table_name.lower():
        return "Customer table essential for contact information retrieval"
    
    if "customer" in priority_entities and "customer" in table_name.lower():
        return "Customer master data table matches primary business entity"
    
    return "Table matched targeted search criteria"

def _explain_column_business_relevance(column_name: str, business_strategy: Dict[str, Any]) -> str:
    """Explain why a column is relevant to the business strategy."""
    primary_intent = business_strategy.get("primary_intent", "")
    col_lower = column_name.lower()
    
    if primary_intent == "contact_lookup":
        if "email" in col_lower:
            return "Email column directly addresses contact lookup requirement"
        elif any(word in col_lower for word in ["first", "last", "name"]):
            return "Name column needed for customer identification"
    
    if "id" in col_lower:
        return "ID column important for data relationships and joins"
    
    return "Column matched targeted search criteria"

def _explain_relationship_business_purpose(from_table: str, to_table: str, business_strategy: Dict[str, Any]) -> str:
    """Explain the business purpose of a relationship."""
    if "customer" in from_table.lower() and "move" in to_table.lower():
        return "Customer-to-moves relationship enables customer activity analysis"
    elif "move" in from_table.lower() and "customer" in to_table.lower():
        return "Moves-to-customer relationship provides customer context for operations"
    else:
        return f"Relationship connects {from_table} and {to_table} for integrated analysis"

def _get_optimized_fallback_schema(query: str) -> str:
    """Provide optimized fallback schema when Weaviate is unavailable."""
    query_lower = query.lower()
    
    # Determine primary intent for optimized fallback
    if any(word in query_lower for word in ['email', 'contact', 'phone']):
        primary_intent = "contact_lookup"
    elif any(word in query_lower for word in ['customer', 'client', 'name']):
        primary_intent = "customer_analysis"
    else:
        primary_intent = "general"
    
    datasets = []
    columns_by_dataset = {}
    
    # Customer table (most common need)
    if primary_intent in ["contact_lookup", "customer_analysis"] or any(word in query_lower for word in ['customer', 'ashraf', 'name']):
        customer_table = {
            "table_name": "customer",
            "athena_table_name": "amspoc3test.customer", 
            "relevance": 0.95,
            "description": "Customer master data (optimized fallback)",
            "record_count": 50000,
            "business_relevance": "Primary customer data source for contact and identification queries"
        }
        datasets.append(customer_table)
        
        # Only include columns that definitely exist
        customer_columns = [
            {
                "column_name": "ID",
                "data_type": "INTEGER",
                "description": "Customer primary key",
                "relevance": 0.90,
                "is_primary_key": True,
                "is_foreign_key": False
            },
            {
                "column_name": "FirstName", 
                "data_type": "VARCHAR",
                "description": "Customer first name",
                "relevance": 0.85,
                "is_primary_key": False,
                "is_foreign_key": False
            },
            {
                "column_name": "LastName",
                "data_type": "VARCHAR", 
                "description": "Customer last name",
                "relevance": 0.85,
                "is_primary_key": False,
                "is_foreign_key": False
            }
        ]
        columns_by_dataset["amspoc3test.customer"] = customer_columns
    
    # Add moves table if relevant
    if any(word in query_lower for word in ['move', 'booking', 'order', '2024']):
        moves_table = {
            "table_name": "moves",
            "athena_table_name": "amspoc3test.moves",
            "relevance": 0.90,
            "description": "Moves and bookings data (optimized fallback)",
            "record_count": 75000,
            "business_relevance": "Primary operational data for move-related queries"
        }
        datasets.append(moves_table)
        
        moves_columns = [
            {
                "column_name": "ID",
                "data_type": "INTEGER", 
                "description": "Move primary key",
                "relevance": 0.90,
                "is_primary_key": True,
                "is_foreign_key": False
            },
            {
                "column_name": "CustomerID",
                "data_type": "INTEGER",
                "description": "Foreign key to customer",
                "relevance": 0.95,
                "is_primary_key": False,
                "is_foreign_key": True
            },
            {
                "column_name": "BookedDate",
                "data_type": "VARCHAR",
                "description": "Booking date (requires casting)",
                "relevance": 0.80,
                "is_primary_key": False,
                "is_foreign_key": False
            }
        ]
        columns_by_dataset["amspoc3test.moves"] = moves_columns
    
    relationships = []
    if len(datasets) > 1:
        relationships.append({
            "from_table": "moves",
            "from_column": "CustomerID", 
            "to_table": "customer",
            "to_column": "ID",
            "join_type": "LEFT JOIN",
            "business_purpose": "Links moves to customer information"
        })
    
    return json.dumps({
        "query": query,
        "business_strategy_applied": {
            "approach": "optimized_fallback",
            "primary_intent": primary_intent,
            "fallback_reason": "Weaviate not available - using optimized schema prediction"
        },
        "datasets": datasets,
        "columns_by_dataset": columns_by_dataset,
        "relationships": relationships,
        "optimization_metrics": {
            "queries_executed": 0,
            "results_filtered": 0,
            "efficiency_score": 0.7,  # Reasonable fallback score
            "fallback_mode": True
        },
        "warning": "🚨 OPTIMIZED FALLBACK MODE - Connect to Weaviate for maximum accuracy"
    }, indent=2)


# Clean up on module exit
import atexit
atexit.register(WeaviateClientSingleton.close)

# --- Legacy Simple Business Context (for backward compatibility) ---
@tool("Simple Business Context")
def analyze_business_context(query: str) -> str:
    """
    Simple business context analysis for Claude 4.
    No hardcoded patterns - pure AI analysis.
    (Kept for backward compatibility)
    """
    logger.info(f"🧠 Analyzing business context: '{query}'")
    
    # Simple context for Claude 4 to analyze
    context = {
        "query": query,
        "domain": "moving_services_company",
        "available_tables": ["customer", "moves", "move_daily"],
        "analysis_needed": "intent_classification_and_domain_mapping"
    }
    
    return json.dumps(context, indent=2)

# --- Intelligent Search Helper Functions ---

def _extract_business_context_from_query(query: str) -> Dict[str, Any]:
    """Extract business context from the query string."""
    try:
        # Check if the query contains business context JSON
        if "business_analysis" in query:
            # Try to extract JSON from the query
            import re
            json_match = re.search(r'\{.*"business_analysis".*\}', query, re.DOTALL)
            if json_match:
                context_data = json.loads(json_match.group())
                return {
                    "intent": context_data.get("business_analysis", {}).get("query_intent", ""),
                    "entities": context_data.get("business_analysis", {}).get("key_entities", []),
                    "domain": context_data.get("business_analysis", {}).get("domain_context", ""),
                    "metrics": context_data.get("business_analysis", {}).get("business_metrics_needed", []),
                    "temporal": context_data.get("business_analysis", {}).get("temporal_requirements", "")
                }
    except:
        pass
    
    # Fallback: analyze the raw query
    query_lower = query.lower()
    entities = []
    
    if any(word in query_lower for word in ['customer', 'client']):
        entities.append("customer")
    if any(word in query_lower for word in ['move', 'booking', 'relocation']):
        entities.append("moves")
    if any(word in query_lower for word in ['email', 'contact', 'phone']):
        entities.append("contact_information")
    if any(word in query_lower for word in ['date', 'time', 'when', '2024', '2023']):
        entities.append("temporal_data")
    
    return {
        "intent": "data_retrieval",
        "entities": entities,
        "domain": "moving_services",
        "metrics": [],
        "temporal": "2024" if "2024" in query_lower else ""
    }

def _create_intelligent_search_queries(query: str, business_context: Dict[str, Any]) -> List[str]:
    """Create multiple targeted search queries based on business context."""
    search_queries = [query]  # Always include original query
    
    # Add business entity-specific queries
    entities = business_context.get("entities", [])
    
    if "customer" in entities:
        search_queries.extend([
            "customer master data CRM contact information",
            "customer table customer_id first_name last_name",
            "customer demographics contact details"
        ])
    
    if "moves" in entities:
        search_queries.extend([
            "moves bookings relocations service orders",
            "move_table booking_date status completion",
            "moving services operations workflow"
        ])
    
    if "contact_information" in entities:
        search_queries.extend([
            "email address phone contact communication",
            "customer contact information email phone",
            "contact details communication preferences"
        ])
    
    # Add domain-specific queries
    domain = business_context.get("domain", "")
    if "moving" in domain:
        search_queries.extend([
            "moving company logistics operations",
            "relocation services customer management"
        ])
    
    return list(set(search_queries))  # Remove duplicates

def _create_column_search_queries(query: str, business_context: Dict[str, Any]) -> List[str]:
    """Create targeted column search queries based on business context."""
    column_queries = [query]
    
    entities = business_context.get("entities", [])
    
    if "customer" in entities:
        column_queries.extend([
            "customer_id first_name last_name customer_name",
            "customer identification name email phone",
            "customer demographics contact information"
        ])
    
    if "contact_information" in entities:
        column_queries.extend([
            "email address email_address contact_email",
            "phone number phone_number contact_phone",
            "contact information communication details"
        ])
    
    if "moves" in entities:
        column_queries.extend([
            "move_id booking_date scheduled_date completion_date",
            "move_status booking_status service_status",
            "origin destination address location"
        ])
    
    if "temporal_data" in entities:
        column_queries.extend([
            "date time timestamp created_date updated_date",
            "booking_date completion_date scheduled_date",
            "datetime temporal date_column"
        ])
    
    return list(set(column_queries))

def _explain_business_relevance(table_name: str, business_context: Dict[str, Any]) -> str:
    """Explain why a table is relevant to the business context."""
    entities = business_context.get("entities", [])
    intent = business_context.get("intent", "")
    
    explanations = []
    
    if "customer" in table_name.lower() and "customer" in entities:
        explanations.append("Contains customer master data relevant to customer-related queries")
    
    if "move" in table_name.lower() and "moves" in entities:
        explanations.append("Contains move/booking data essential for operational queries")
    
    if "contact" in intent.lower() and "customer" in table_name.lower():
        explanations.append("Customer table likely contains contact information")
    
    if not explanations:
        explanations.append("Table matched semantic search criteria")
    
    return "; ".join(explanations)

def _explain_column_business_relevance(column_name: str, business_context: Dict[str, Any]) -> str:
    """Explain why a column is relevant to the business context."""
    entities = business_context.get("entities", [])
    intent = business_context.get("intent", "").lower()
    
    col_lower = column_name.lower()
    
    if "email" in col_lower and "contact_information" in entities:
        return "Email column directly matches contact information requirement"
    
    if "customer" in col_lower and "customer" in entities:
        return "Customer-related column essential for customer queries"
    
    if "name" in col_lower and "customer" in entities:
        return "Name column needed for customer identification"
    
    if "date" in col_lower and business_context.get("temporal"):
        return "Date column relevant for temporal filtering"
    
    if "id" in col_lower:
        return "ID column important for joins and data relationships"
    
    return "Column matched semantic search criteria"

def _get_table_combinations(table_names: List[str]) -> List[tuple]:
    """Get all combinations of tables for relationship discovery."""
    combinations = []
    for i, table1 in enumerate(table_names):
        for table2 in table_names[i+1:]:
            combinations.append((table1, table2))
    return combinations

def _explain_relationship_purpose(from_table: str, to_table: str, business_context: Dict[str, Any]) -> str:
    """Explain the business purpose of a relationship."""
    if "customer" in from_table.lower() and "move" in to_table.lower():
        return "Links customers to their moves/bookings for comprehensive customer view"
    elif "move" in from_table.lower() and "customer" in to_table.lower():
        return "Links moves/bookings to customer information for service delivery"
    else:
        return f"Connects {from_table} and {to_table} data for integrated analysis"

def _generate_business_approach_recommendation(results: Dict[str, Any], business_context: Dict[str, Any]) -> str:
    """Generate business-focused approach recommendations."""
    entities = business_context.get("entities", [])
    intent = business_context.get("intent", "")
    
    if "contact" in intent.lower() and "customer" in entities:
        return "Use customer table with name-based filtering to retrieve contact information"
    
    if "customer" in entities and "moves" in entities:
        return "Join customer and moves tables to analyze customer activity and service delivery"
    
    if business_context.get("temporal"):
        return "Focus on temporal filtering and date-based analysis for time-series insights"
    
    return "Use discovered tables and relationships for comprehensive data analysis"

def _assess_data_quality_for_business_use(results: Dict[str, Any]) -> List[str]:
    """Assess data quality considerations for business use."""
    insights = []
    
    datasets = results.get("datasets", [])
    if datasets:
        for dataset in datasets:
            record_count = dataset.get("record_count", 0)
            if record_count < 100:
                insights.append(f"Small dataset size ({record_count} records) in {dataset['table_name']} - verify data completeness")
    
    columns = results.get("columns_by_dataset", {})
    for table, cols in columns.items():
        varchar_dates = [c for c in cols if c.get("data_type") == "VARCHAR" and "date" in c.get("column_name", "").lower()]
        if varchar_dates:
            insights.append(f"Date columns stored as VARCHAR in {table} - requires casting for date operations")
    
    if not insights:
        insights.append("No obvious data quality issues detected")
    
    return insights

def _generate_performance_tips(results: Dict[str, Any]) -> List[str]:
    """Generate performance optimization tips."""
    tips = []
    
    datasets = results.get("datasets", [])
    if len(datasets) > 2:
        tips.append("Multiple tables involved - consider query complexity and join optimization")
    
    columns = results.get("columns_by_dataset", {})
    for table, cols in columns.items():
        varchar_dates = [c for c in cols if c.get("data_type") == "VARCHAR" and "date" in c.get("column_name", "").lower()]
        if varchar_dates:
            tips.append(f"Use TRY_CAST for date conversion in {table} to handle invalid date formats")
    
    relationships = results.get("relationships", [])
    if len(relationships) > 3:
        tips.append("Multiple joins detected - verify join conditions and consider query performance impact")
    
    if not tips:
        tips.append("Use standard Athena optimization practices for best performance")
    
    return tips

def _get_intelligent_fallback_schema(query: str) -> str:
    """Provide intelligent fallback schema when Weaviate is unavailable."""
    query_lower = query.lower()
    
    datasets = []
    columns_by_dataset = {}
    relationships = []
    
    # Always include moves table
    moves_table = {
        "table_name": "moves",
        "athena_table_name": "amspoc3test.moves",
        "relevance": 0.95,
        "description": "Primary moves and booking data (fallback schema)",
        "record_count": 50000
    }
    datasets.append(moves_table)
    
    # Standard moves columns (only real columns)
    moves_columns = [
        {
            "column_name": "ID",
            "data_type": "INTEGER",
            "description": "Primary key for moves table",
            "relevance": 0.90,
            "is_primary_key": True,
            "is_foreign_key": False
        },
        {
            "column_name": "CustomerID",
            "data_type": "INTEGER",
            "description": "Customer identifier for joins",
            "relevance": 0.95,
            "is_primary_key": False,
            "is_foreign_key": True
        },
        {
            "column_name": "BookedDate",
            "data_type": "VARCHAR",
            "description": "Date when move was booked",
            "relevance": 0.90,
            "is_primary_key": False,
            "is_foreign_key": False
        },
        {
            "column_name": "Status",
            "data_type": "VARCHAR",
            "description": "Move status",
            "relevance": 0.85,
            "is_primary_key": False,
            "is_foreign_key": False
        }
    ]
    columns_by_dataset["amspoc3test.moves"] = moves_columns
    
    # Add customer table if query mentions customers
    if any(word in query_lower for word in ['customer', 'client', 'name', 'contact']):
        customer_table = {
            "table_name": "customer",
            "athena_table_name": "amspoc3test.customer",
            "relevance": 0.85,
            "description": "Customer master data (fallback schema)",
            "record_count": 15000
        }
        datasets.append(customer_table)
        
        # Only use columns that actually exist
        customer_columns = [
            {
                "column_name": "ID",
                "data_type": "INTEGER",
                "description": "Customer primary key",
                "relevance": 0.90,
                "is_primary_key": True,
                "is_foreign_key": False
            },
            {
                "column_name": "FirstName",
                "data_type": "VARCHAR",
                "description": "Customer first name",
                "relevance": 0.85,
                "is_primary_key": False,
                "is_foreign_key": False
            },
            {
                "column_name": "LastName",
                "data_type": "VARCHAR",
                "description": "Customer last name",
                "relevance": 0.85,
                "is_primary_key": False,
                "is_foreign_key": False
            }
        ]
        columns_by_dataset["amspoc3test.customer"] = customer_columns
        
        relationships.append({
            "from_table": "moves",
            "from_column": "CustomerID",
            "to_table": "customer",
            "to_column": "ID",
            "join_type": "LEFT JOIN"
        })
    
    return json.dumps({
        "query": query,
        "datasets": datasets,
        "columns_by_dataset": columns_by_dataset,
        "relationships": relationships,
        "fallback_mode": True,
        "warning": "🚨 USING FALLBACK SCHEMA - Connect to Weaviate for accurate schema discovery",
        "business_intelligence_summary": {
            "primary_business_intent": "Fallback mode - limited business intelligence",
            "key_business_entities": ["moves", "customer"] if "customer" in query_lower else ["moves"],
            "recommended_approach": "Verify column names against actual database schema before execution"
        }
    }, indent=2)