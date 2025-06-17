# tools/athena_tools.py - COMPLETE UPDATED VERSION WITH FULL METADATA UTILIZATION
# Prevents hallucination by using ALL Weaviate metadata for SQL generation

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
import boto3
from dotenv import load_dotenv
from crewai.tools import tool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveAIQueryEngine:
    """
    COMPREHENSIVE AI-driven SQL query engine that uses ALL Weaviate metadata
    - Prevents hallucination by strictly constraining to discovered schema
    - Uses semantic types, sample values, business context, and proven patterns
    - Leverages answerable_questions and llm_hints for optimization
    - Applies data_quirks and relationship information
    """
    
    def __init__(self):
        """Initialize with proper validation and environment setup"""
        
        # Validate critical environment variables
        required_env_vars = {
            'AWS_REGION': os.getenv('AWS_REGION'),
            'ATHENA_DATABASE': os.getenv('ATHENA_DATABASE'),
            'ATHENA_OUTPUT_LOCATION': os.getenv('ATHENA_OUTPUT_LOCATION')
        }
        
        missing_vars = [key for key, value in required_env_vars.items() if not value]
        if missing_vars:
            logger.warning(f"⚠️ Missing critical environment variables: {missing_vars}")
        
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            # FIXED: Correct model ID without 'us.' prefix
            self.model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'anthropic.claude-3-5-haiku-20241022-v1:0')
            
            # Test Bedrock connection
            try:
                self.bedrock_client.list_foundation_models(maxResults=1)
                logger.info("✅ Bedrock client initialized and tested successfully")
            except Exception as e:
                logger.warning(f"⚠️ Bedrock connection test failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Bedrock client: {e}")
            raise
        
        # Initialize caching and configuration
        self._sql_cache = {}
        self._max_cache_size = 100
        self.fallback_tables = os.getenv('FALLBACK_TABLES', 'amspoc3test.customer,amspoc3test.moves').split(',')
        
        logger.info(f"🔧 Configured fallback tables: {self.fallback_tables}")
    
    def generate_comprehensive_sql(self, question: str, blueprint: Dict) -> str:
        """
        Generate SQL using COMPLETE Weaviate metadata to prevent hallucination
        
        Args:
            question: Original business question
            blueprint: Complete adaptive blueprint with all metadata
            
        Returns:
            AI-generated SQL query using only discovered schema elements
        """
        logger.info(f"🧠 Generating comprehensive SQL for: {question}")
        
        # STEP 1: Parse and validate complete blueprint structure
        try:
            metadata = self._parse_complete_blueprint(blueprint)
            if not metadata['valid']:
                logger.error(f"❌ Invalid blueprint: {metadata['error']}")
                return self._get_emergency_fallback_query(question)
            
            logger.info(f"✅ Parsed blueprint: {len(metadata['datasets'])} datasets, {len(metadata['columns_by_dataset'])} tables")
            
        except Exception as e:
            logger.error(f"❌ Blueprint parsing failed: {e}")
            return self._get_emergency_fallback_query(question)
        
        # STEP 2: Check cache for identical requests
        cache_key = self._generate_cache_key(question, metadata)
        if cache_key in self._sql_cache:
            logger.info("✅ Using cached SQL result")
            return self._sql_cache[cache_key]
        
        # STEP 3: Create comprehensive schema context using ALL metadata
        comprehensive_context = self._create_comprehensive_schema_context(metadata)
        
        # STEP 4: Generate SQL using enriched context
        try:
            sql = self._generate_sql_with_full_metadata(question, comprehensive_context, metadata)
            
            # STEP 5: Validate generated SQL against discovered schema
            validation = self._validate_sql_against_schema(sql, metadata)
            
            if not validation['valid']:
                logger.warning(f"⚠️ Generated SQL failed validation: {validation['errors']}")
                sql = self._get_schema_aware_fallback(question, metadata)
            
            # STEP 6: Cache successful result
            self._cache_sql_result(cache_key, sql)
            
            logger.info("✅ Comprehensive SQL generation completed")
            return sql
            
        except Exception as e:
            logger.error(f"❌ SQL generation failed: {e}")
            return self._get_schema_aware_fallback(question, metadata)
    
    def _parse_complete_blueprint(self, blueprint: Dict) -> Dict:
        """Parse and validate complete blueprint structure"""
        
        metadata = {
            'valid': False,
            'error': '',
            'datasets': [],
            'columns_by_dataset': {},
            'relationships': [],
            'column_groups': {},
            'answerable_questions': [],
            'llm_hints': {},
            'stage_1_context': {}
        }
        
        try:
            # Handle string input
            if isinstance(blueprint, str):
                blueprint = json.loads(blueprint)
            
            # Extract Stage 1 context for business intelligence
            stage_1 = blueprint.get('stage_1_adaptive_context', {})
            metadata['stage_1_context'] = {
                'original_query': stage_1.get('original_query', ''),
                'triage_result': stage_1.get('triage_result', {}),
                'contextual_warnings': stage_1.get('contextual_warnings', []),
                'enrichment_sources': stage_1.get('enrichment_sources', {})
            }
            
            # Extract Stage 2 precision discovery
            stage_2 = blueprint.get('stage_2_precision_discovery', {})
            
            if not stage_2:
                metadata['error'] = 'Missing stage_2_precision_discovery'
                return metadata
            
            # Extract all components
            metadata['datasets'] = stage_2.get('datasets', [])
            metadata['columns_by_dataset'] = stage_2.get('columns_by_dataset', {})
            metadata['relationships'] = stage_2.get('relationships', [])
            metadata['column_groups'] = stage_2.get('column_groups_discovered', {})
            metadata['answerable_questions'] = stage_2.get('answerable_questions_matched', [])
            metadata['llm_hints'] = stage_2.get('llm_hints_discovered', {})
            
            # Validate essential components
            if not metadata['columns_by_dataset']:
                metadata['error'] = 'No columns_by_dataset found'
                return metadata
            
            if not metadata['datasets']:
                metadata['error'] = 'No datasets found'
                return metadata
            
            metadata['valid'] = True
            logger.info(f"📊 Blueprint contains: {len(metadata['datasets'])} datasets, {len(metadata['relationships'])} relationships, {len(metadata['answerable_questions'])} proven questions")
            
            return metadata
            
        except Exception as e:
            metadata['error'] = f'Blueprint parsing exception: {str(e)}'
            return metadata
    
    def _create_comprehensive_schema_context(self, metadata: Dict) -> str:
        """Create comprehensive schema context using ALL Weaviate metadata"""
        
        context_sections = []
        
        # SECTION 1: Business Context from Stage 1
        stage_1 = metadata['stage_1_context']
        if stage_1:
            context_sections.append("BUSINESS CONTEXT:")
            context_sections.append(f"Original Query: {stage_1.get('original_query', 'Unknown')}")
            
            triage = stage_1.get('triage_result', {})
            if triage:
                context_sections.append(f"Query Complexity: {triage.get('complexity', 'Unknown')}")
                context_sections.append(f"Key Entities: {', '.join(triage.get('key_entities', []))}")
                context_sections.append(f"Domain Context: {triage.get('domain_context', 'Unknown')}")
            
            warnings = stage_1.get('contextual_warnings', [])
            if warnings:
                context_sections.append(f"Data Quality Warnings: {'; '.join(warnings)}")
        
        # SECTION 2: Available Tables with Business Intelligence
        context_sections.append("\nAVAILABLE TABLES (MUST USE ONLY THESE):")
        for i, dataset in enumerate(metadata['datasets'], 1):
            table_name = dataset.get('table_name', 'unknown')
            athena_name = dataset.get('athena_table_name', table_name)
            relevance = dataset.get('relevance_score', 0)
            description = dataset.get('description', '')
            business_purpose = dataset.get('business_purpose', '')
            data_owner = dataset.get('data_owner', '')
            tags = dataset.get('tags', [])
            
            context_sections.append(f"""
{i}. TABLE: {athena_name}
   Original Name: {table_name}
   Relevance Score: {relevance:.3f}
   Description: {description[:200]}{'...' if len(description) > 200 else ''}
   Business Purpose: {business_purpose[:150]}{'...' if len(business_purpose) > 150 else ''}
   Data Owner: {data_owner}
   Tags: {', '.join(tags)}
   SQL Usage: SELECT ... FROM {athena_name}
        """)
        
        # SECTION 3: Comprehensive Column Information with ALL Schema Metadata
        context_sections.append("\nAVAILABLE COLUMNS (MUST USE ONLY THESE):")
        for table_name, columns in metadata['columns_by_dataset'].items():
            context_sections.append(f"\n🎯 TABLE: {table_name}")
            
            # Sort columns by AI relevance score
            sorted_columns = sorted(columns, key=lambda x: x.get('ai_relevance_score', 0), reverse=True)
            
            context_sections.append("   COLUMNS (ordered by AI relevance):")
            for col in sorted_columns:
                col_name = col.get('column_name', 'unknown')
                athena_data_type = col.get('athena_data_type', 'string')
                semantic_type = col.get('semantic_type', '')
                business_name = col.get('business_name', '')
                data_classification = col.get('data_classification', '')
                description = col.get('description', '')
                sample_values = col.get('sample_values', [])
                is_pk = col.get('is_primary_key', False)
                is_fk = col.get('is_foreign_key', False)
                ai_score = col.get('ai_relevance_score', 0)
                column_group = col.get('column_group', '')
                
                # 🔥 RICH METADATA FROM SCHEMA (newly added)
                aggregation_patterns = col.get('aggregation_patterns', [])
                common_filters = col.get('common_filters', [])
                sql_usage_pattern = col.get('sql_usage_pattern', '')
                usage_hints = col.get('usage_hints', [])
                foreign_key_target_table = col.get('foreign_key_target_table', '')
                foreign_key_target_column = col.get('foreign_key_target_column', '')
                join_pattern = col.get('join_pattern', '')
                
                # Create priority indicator
                if ai_score >= 9.0:
                    priority = "🔥 CRITICAL"
                elif ai_score >= 8.0:
                    priority = "⭐ HIGH"
                elif ai_score >= 7.0:
                    priority = "📋 MEDIUM"
                else:
                    priority = "📌 LOW"
                
                # Build comprehensive column info
                column_info = f"""
    • {col_name} ({athena_data_type}) - {priority} [AI Score: {ai_score:.1f}]
      Business Name: {business_name}
      Semantic Type: {semantic_type} | Group: {column_group}
      Description: {description[:100]}{'...' if len(description) > 100 else ''}
      Classification: {data_classification}
      Primary Key: {is_pk} | Foreign Key: {is_fk}
      Sample Values: {sample_values[:5] if sample_values else 'None'}"""
                
                # Add rich schema metadata
                if sql_usage_pattern:
                    column_info += f"\n     SQL Usage Pattern: {sql_usage_pattern}"
                
                if usage_hints:
                    column_info += f"\n     Usage Hints: {', '.join(usage_hints)}"
                
                if aggregation_patterns:
                    column_info += f"\n     Proven Aggregations: {', '.join(aggregation_patterns[:3])}"  # Top 3
                
                if common_filters:
                    column_info += f"\n     Common Filters: {', '.join(common_filters[:2])}"  # Top 2
                
                if is_fk and foreign_key_target_table:
                    column_info += f"\n     Foreign Key: → {foreign_key_target_table}.{foreign_key_target_column}"
                    if join_pattern:
                        column_info += f"\n     Join Pattern: {join_pattern}"
                
                column_info += f"\n     SQL Reference: SELECT {col_name} FROM {table_name}"
                
                context_sections.append(column_info)
        
        # SECTION 4: Relationship Information for JOINs
        relationships = metadata.get('relationships', [])
        if relationships:
            context_sections.append("\nAVAILABLE RELATIONSHIPS (USE FOR JOINS):")
            for i, rel in enumerate(relationships, 1):
                from_table = rel.get('from_table', '')
                from_col = rel.get('from_column', '')
                to_table = rel.get('to_table', '')
                to_col = rel.get('to_column', '')
                join_type = rel.get('join_type', 'LEFT JOIN')
                discovered_via = rel.get('discovered_via', '')
                fk_definition = rel.get('foreign_key_definition', '')
                
                context_sections.append(f"""
{i}. RELATIONSHIP: {from_table}.{from_col} → {to_table}.{to_col}
   Join Type: {join_type}
   Discovered Via: {discovered_via}
   Definition: {fk_definition}
   SQL Syntax: {join_type} {to_table} ON {from_table}.{from_col} = {to_table}.{to_col}
            """)
        
        # SECTION 5: Column Groups for Logical Selection
        column_groups = metadata.get('column_groups', {})
        if column_groups:
            context_sections.append("\nCOLUMN GROUPS (select related columns together):")
            for group_name, group_columns in column_groups.items():
                context_sections.append(f"   {group_name}: {', '.join(group_columns)}")
        
        # SECTION 6: Proven Query Patterns from answerableQuestions
        answerable_questions = metadata.get('answerable_questions', [])
        if answerable_questions:
            context_sections.append("\nPROVEN QUERY PATTERNS (adapt these working examples):")
            for i, q in enumerate(answerable_questions, 1):
                question = q.get('question', '')
                sql_hint = q.get('sql_hint', '')
                category = q.get('category', '')
                relevance = q.get('relevance_to_query', 0)
                
                context_sections.append(f"""
{i}. PATTERN: {question}
   Category: {category} | Relevance: {relevance:.2f}
   Proven SQL: {sql_hint}
            """)
        
        # SECTION 7: LLM Optimization Hints (ENHANCED with Schema Data)
        llm_hints = metadata.get('llm_hints', {})
        if llm_hints:
            context_sections.append("\nOPTIMIZATION HINTS FROM SCHEMA:")
            
            # Preferred aggregations (from schema)
            aggregations = llm_hints.get('preferred_aggregations', [])
            if aggregations:
                context_sections.append("   📊 Preferred Aggregations (from schema):")
                for agg in aggregations:
                    context_sections.append(f"     - {agg}")
            
            # Common filters (from schema)
            filters = llm_hints.get('common_filters', [])
            if filters:
                context_sections.append("   🔍 Common Filters (from schema):")
                for filt in filters:
                    context_sections.append(f"     - {filt}")
            
            # Join patterns (from schema)
            join_patterns = llm_hints.get('join_patterns', [])
            if join_patterns:
                context_sections.append("   🔗 Join Patterns (from schema):")
                for pattern in join_patterns:
                    context_sections.append(f"     - {pattern}")
            
            # Data quirks (CRITICAL for avoiding errors)
            data_quirks = llm_hints.get('data_quirks', [])
            if data_quirks:
                context_sections.append("   🚨 DATA QUIRKS FROM SCHEMA (MUST HANDLE):")
                for quirk in data_quirks:
                    context_sections.append(f"     ⚠️ {quirk}")
        
        # SECTION 8: Critical Constraints (Enhanced)
        available_tables = list(metadata['columns_by_dataset'].keys())
        context_sections.append(f"""
🚨 CRITICAL CONSTRAINTS:
1. NEVER use tables not in this list: {available_tables}
2. NEVER use columns not listed in the schema above
3. ALWAYS use exact athena_data_type from schema
4. ALWAYS apply data_quirks handling from schema
5. PRIORITIZE columns with highest AI relevance scores
6. USE semantic_type information for appropriate functions
7. APPLY sample_values for format-aware filtering
8. FOLLOW proven query patterns when applicable
9. USE relationship information for accurate JOINs
10. LEVERAGE aggregation_patterns from schema for GROUP BY
11. APPLY common_filters from schema for WHERE clauses
12. FOLLOW sql_usage_pattern from schema for formatting
13. USE usage_hints from schema for column-specific guidance
14. INCLUDE appropriate WHERE clauses and LIMIT clause
15. RESPECT data_classification for security compliance
    """)
        
        return "\n".join(context_sections)
    
    def _generate_sql_with_full_metadata(self, question: str, comprehensive_context: str, metadata: Dict) -> str:
        """Generate SQL using comprehensive metadata context"""
        
        # Extract key information for prompt optimization
        available_tables = list(metadata['columns_by_dataset'].keys())
        high_value_columns = self._extract_high_value_columns(metadata)
        proven_patterns = self._extract_proven_patterns(metadata, question)
        critical_quirks = self._extract_critical_quirks(metadata)
        
        # Create focused prompt with comprehensive context
        prompt = f"""You are an expert AWS Athena SQL generator. Generate SQL to answer this question using ONLY the provided schema.

QUESTION: "{question}"

{comprehensive_context}

GENERATION STRATEGY:
1. PRIORITIZE columns with AI relevance score >= 9.0 for main SELECT
2. USE semantic_type to choose appropriate functions:
   - financial_amount → SUM(), ROUND(), numeric aggregations
   - business_date → TRY_CAST AS DATE, date functions
   - status_code → categorical filtering, COUNT by status
   - identifier → use in JOINs and WHERE clauses
3. APPLY sample_values for format-aware filtering
4. ADAPT proven query patterns when relevant: {proven_patterns}
5. HANDLE data quirks: {critical_quirks}
6. USE exact table names: {available_tables}
7. FOLLOW relationship patterns for JOINs

HIGH-VALUE COLUMNS TO PRIORITIZE:
{high_value_columns}

ATHENA BEST PRACTICES:
- Use TRY_CAST for all type conversions
- Include LIMIT clause (default 100)
- Use proper WHERE clauses for filtering
- Apply NULL checks where appropriate
- Use exact column names and data types from schema

Return ONLY the SQL query, no explanation or markdown formatting."""
        
        return self._invoke_bedrock_with_retries(prompt)
    
    def _extract_high_value_columns(self, metadata: Dict) -> str:
        """Extract highest value columns for prioritization"""
        high_value = []
        
        for table_name, columns in metadata['columns_by_dataset'].items():
            sorted_cols = sorted(columns, key=lambda x: x.get('ai_relevance_score', 0), reverse=True)
            
            for col in sorted_cols[:3]:  # Top 3 per table
                col_name = col.get('column_name', '')
                ai_score = col.get('ai_relevance_score', 0)
                semantic_type = col.get('semantic_type', '')
                
                high_value.append(f"{table_name}.{col_name} (Score: {ai_score:.1f}, Type: {semantic_type})")
        
        return "\n".join(high_value)
    
    def _extract_proven_patterns(self, metadata: Dict, question: str) -> str:
        """Extract most relevant proven query patterns"""
        patterns = []
        
        answerable_questions = metadata.get('answerable_questions', [])
        question_lower = question.lower()
        
        # Find most relevant patterns
        for q in answerable_questions:
            q_text = q.get('question', '').lower()
            sql_hint = q.get('sql_hint', '')
            
            # Simple relevance matching
            if any(word in q_text for word in question_lower.split() if len(word) > 3):
                patterns.append(f"Pattern: {sql_hint}")
        
        return "; ".join(patterns[:2])  # Top 2 patterns
    
    def _extract_critical_quirks(self, metadata: Dict) -> str:
        """Extract critical data quirks that must be handled"""
        quirks = []
        
        llm_hints = metadata.get('llm_hints', {})
        data_quirks = llm_hints.get('data_quirks', [])
        
        # Add contextual warnings
        stage_1 = metadata.get('stage_1_context', {})
        warnings = stage_1.get('contextual_warnings', [])
        
        quirks.extend(data_quirks)
        quirks.extend(warnings)
        
        return "; ".join(quirks[:3])  # Top 3 critical quirks
    
    def _invoke_bedrock_with_retries(self, prompt: str, max_retries: int = 3) -> str:
        """Invoke Bedrock with comprehensive error handling and retries"""
        
        for attempt in range(max_retries):
            try:
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps({
                        "messages": [{"role": "user", "content": prompt}],
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1500,
                        "temperature": 0.1  # Low temperature for consistent SQL
                    })
                )
                
                result = json.loads(response['body'].read())
                
                # Validate response structure
                if 'content' not in result or not result['content']:
                    raise ValueError("Invalid Bedrock response structure")
                
                content = result['content'][0]['text'].strip()
                
                if not content:
                    raise ValueError("Empty response from Bedrock")
                
                # Clean up formatting
                content = content.replace('```sql', '').replace('```SQL', '').replace('```', '').strip()
                
                # Basic SQL validation
                if 'select' not in content.lower():
                    raise ValueError("Generated content does not contain valid SQL")
                
                logger.info(f"✅ Bedrock SQL generation successful on attempt {attempt + 1}")
                return content
                
            except Exception as e:
                logger.warning(f"❌ Bedrock attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("All Bedrock retry attempts failed")
    
    def _validate_sql_against_schema(self, sql: str, metadata: Dict) -> Dict:
        """Validate generated SQL against discovered schema"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'tables_used': [],
            'columns_referenced': []
        }
        
        sql_lower = sql.lower()
        available_tables = list(metadata['columns_by_dataset'].keys())
        
        # Check basic SQL structure
        if 'select' not in sql_lower:
            validation['valid'] = False
            validation['errors'].append("Missing SELECT statement")
        
        if 'from' not in sql_lower:
            validation['valid'] = False
            validation['errors'].append("Missing FROM clause")
        
        # Check table usage
        for table_name in available_tables:
            table_simple = table_name.split('.')[-1].lower()
            if table_name.lower() in sql_lower or table_simple in sql_lower:
                validation['tables_used'].append(table_name)
        
        if not validation['tables_used']:
            validation['valid'] = False
            validation['errors'].append("No recognized tables found in SQL")
        
        # Check for prohibited tables (common hallucinations)
        prohibited_tables = ['sales_dataset', 'daily_operations', 'sample_table', 'test_table']
        for prohibited in prohibited_tables:
            if prohibited in sql_lower:
                validation['valid'] = False
                validation['errors'].append(f"Prohibited/hallucinated table '{prohibited}' found")
        
        # Check for LIMIT clause
        if 'limit' not in sql_lower:
            validation['warnings'].append("Missing LIMIT clause - may impact performance")
        
        return validation
    
    def _generate_cache_key(self, question: str, metadata: Dict) -> str:
        """Generate cache key for SQL caching"""
        # Create a hash of question + available tables/columns
        tables = list(metadata['columns_by_dataset'].keys())
        column_signature = {}
        
        for table, columns in metadata['columns_by_dataset'].items():
            column_signature[table] = [col.get('column_name', '') for col in columns]
        
        cache_data = {
            'question': question.lower().strip(),
            'tables': sorted(tables),
            'columns': column_signature
        }
        
        return str(hash(json.dumps(cache_data, sort_keys=True)))
    
    def _cache_sql_result(self, cache_key: str, sql: str) -> None:
        """Cache SQL result with size management"""
        if len(self._sql_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._sql_cache))
            del self._sql_cache[oldest_key]
        
        self._sql_cache[cache_key] = sql
        logger.debug(f"📦 Cached SQL result (cache size: {len(self._sql_cache)})")
    
    def _get_schema_aware_fallback(self, question: str, metadata: Dict) -> str:
        """Generate fallback query using actual schema metadata"""
        
        if not metadata['columns_by_dataset']:
            return self._get_emergency_fallback_query(question)
        
        # Get highest relevance table and columns
        best_table = None
        best_columns = []
        highest_relevance = 0
        
        for table_name, columns in metadata['columns_by_dataset'].items():
            # Calculate table relevance based on column AI scores
            avg_relevance = sum(col.get('ai_relevance_score', 0) for col in columns) / len(columns)
            
            if avg_relevance > highest_relevance:
                highest_relevance = avg_relevance
                best_table = table_name
                best_columns = sorted(columns, key=lambda x: x.get('ai_relevance_score', 0), reverse=True)
        
        if not best_table:
            return self._get_emergency_fallback_query(question)
        
        # Generate schema-aware fallback
        question_lower = question.lower()
        
        # Select appropriate columns based on question intent and AI scores
        selected_columns = []
        
        # Always include primary key if available
        pk_cols = [col for col in best_columns if col.get('is_primary_key')]
        if pk_cols:
            selected_columns.append(pk_cols[0]['column_name'])
        
        # Add high-scoring columns relevant to question
        for col in best_columns[:5]:  # Top 5 columns
            col_name = col.get('column_name', '')
            semantic_type = col.get('semantic_type', '')
            ai_score = col.get('ai_relevance_score', 0)
            
            if ai_score >= 8.0 and col_name not in selected_columns:
                # Check question relevance
                if any(keyword in col_name.lower() or keyword in semantic_type.lower() 
                      for keyword in question_lower.split() if len(keyword) > 3):
                    selected_columns.append(col_name)
                elif len(selected_columns) < 3:  # Ensure we have at least 3 columns
                    selected_columns.append(col_name)
        
        # Ensure we have at least some columns
        if not selected_columns:
            selected_columns = [col['column_name'] for col in best_columns[:3]]
        
        # Create WHERE clause using primary key or first column
        where_column = selected_columns[0]
        
        return f"""
SELECT {', '.join(selected_columns)}
FROM {best_table}
WHERE {where_column} IS NOT NULL
LIMIT 10
        """.strip()
    
    def _get_emergency_fallback_query(self, question: str) -> str:
        """Emergency fallback when no schema is available"""
        
        primary_table = self.fallback_tables[0].strip() if self.fallback_tables else 'amspoc3test.customer'
        logger.warning(f"🚨 Using emergency fallback with table: {primary_table}")
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['count', 'total', 'how many']):
            return f"SELECT COUNT(*) as total_count FROM {primary_table} LIMIT 1"
        
        elif any(word in question_lower for word in ['revenue', 'cost', 'amount', 'sum']):
            return f"SELECT COUNT(*) as record_count FROM {primary_table} LIMIT 1"
        
        else:
            return f"SELECT * FROM {primary_table} WHERE ID IS NOT NULL LIMIT 5"


class ComprehensiveAnalysisEngine:
    """
    COMPREHENSIVE AI-driven analysis engine that uses blueprint metadata for insights
    """
    
    def __init__(self):
        """Initialize analysis engine"""
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'anthropic.claude-3-5-haiku-20241022-v1:0')
    
    def analyze_results_with_metadata(self, question: str, results: Dict, original_blueprint: Dict = None) -> Dict:
        """
        Analyze query results using original blueprint metadata for enhanced insights
        """
        
        # Prepare comprehensive analysis context
        results_context = self._prepare_comprehensive_results_context(results)
        blueprint_context = self._extract_blueprint_insights(original_blueprint) if original_blueprint else ""
        
        prompt = f"""You are an expert business data analyst. Analyze these query results and provide comprehensive business insights using the provided metadata context.

ORIGINAL QUESTION: "{question}"

QUERY RESULTS:
{results_context}

BLUEPRINT METADATA CONTEXT:
{blueprint_context}

ANALYSIS REQUIREMENTS:
1. Provide a direct, clear answer to the original question
2. Extract key business insights from the data using metadata context
3. Assess business impact using data owner and business purpose information
4. Identify risks and opportunities based on data patterns
5. Provide strategic recommendations using business context
6. Evaluate data quality using metadata information
7. Assign confidence level based on data completeness and metadata richness

RESPONSE FORMAT (JSON):
{{
    "direct_answer": "Clear, concise answer to the original question",
    "key_insights": [
        "Insight 1 with specific data points and business context",
        "Insight 2 with quantitative details and metadata context",
        "Insight 3 with trend information and strategic implications"
    ],
    "strategic_analysis": {{
        "business_impact": "Assessment using business_purpose and data_owner context",
        "confidence_level": "HIGH/MEDIUM/LOW based on metadata richness",
        "risk_factors": ["Risk 1 based on data patterns", "Risk 2 from metadata context"],
        "opportunities": ["Opportunity 1 from results", "Opportunity 2 from business context"],
        "strategic_recommendations": [
            "Actionable recommendation 1 based on data and metadata",
            "Actionable recommendation 2 using business context"
        ]
    }},
    "metadata_enhanced_insights": {{
        "data_quality_assessment": "Quality evaluation using metadata context",
        "business_context_application": "How business_purpose and data_owner informed analysis",
        "semantic_interpretation": "Analysis using semantic_type and column_group information",
        "relationship_insights": "Cross-table analysis opportunities from relationships"
    }},
    "performance_insights": "Query performance analysis and optimization opportunities"
}}

Return ONLY valid JSON, no additional text."""
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "temperature": 0.3
                })
            )
            
            result = json.loads(response['body'].read())
            ai_analysis = result['content'][0]['text'].strip()
            
            # Extract and parse JSON
            import re
            json_match = re.search(r'\{.*\}', ai_analysis, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_fallback_analysis(question, results)
                
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            return self._get_fallback_analysis(question, results)
    
    def _prepare_comprehensive_results_context(self, results: Dict) -> str:
        """Prepare comprehensive results context for analysis"""
        
        if results.get('status') == 'error':
            error_details = results.get('error', {})
            return f"""
ERROR: Query execution failed
Error Code: {error_details.get('code', 'Unknown')}
Error Message: {error_details.get('message', 'No details')}
Query ID: {error_details.get('query_id', 'Unknown')}
            """
        
        rows = results.get('rows', [])
        columns = results.get('columns', [])
        performance = results.get('performance_metrics', {})
        
        context_parts = [
            f"COLUMNS: {', '.join(columns)}",
            f"ROW COUNT: {len(rows)}",
            f"EXECUTION TIME: {performance.get('execution_time_ms', 0)}ms",
            f"DATA SCANNED: {performance.get('data_scanned_mb', 0)} MB"
        ]
        
        if rows:
            context_parts.append("\nDATA SAMPLE:")
            # Show more comprehensive sample
            for i, row in enumerate(rows[:10]):  # Show up to 10 rows
                row_data = dict(zip(columns, row))
                context_parts.append(f"Row {i+1}: {row_data}")
            
            if len(rows) > 10:
                context_parts.append(f"... and {len(rows) - 10} more rows")
            
            # Add data summary statistics if numeric data present
            context_parts.append("\nDATA SUMMARY:")
            for i, col in enumerate(columns):
                col_values = [row[i] for row in rows if row[i] is not None]
                if col_values:
                    # Try to identify numeric data
                    try:
                        numeric_values = [float(val) for val in col_values if str(val).replace('.', '').replace('-', '').isdigit()]
                        if numeric_values:
                            context_parts.append(f"{col}: Min={min(numeric_values)}, Max={max(numeric_values)}, Avg={sum(numeric_values)/len(numeric_values):.2f}")
                    except:
                        context_parts.append(f"{col}: {len(set(col_values))} unique values")
        
        return "\n".join(context_parts)
    
    def _extract_blueprint_insights(self, blueprint: Dict) -> str:
        """Extract insights from original blueprint for enhanced analysis"""
        
        if not blueprint:
            return "No blueprint metadata available"
        
        insights = []
        
        # Extract Stage 1 business context
        stage_1 = blueprint.get('stage_1_adaptive_context', {})
        if stage_1:
            insights.append("BUSINESS CONTEXT:")
            insights.append(f"Query Complexity: {stage_1.get('triage_result', {}).get('complexity', 'Unknown')}")
            insights.append(f"Domain: {stage_1.get('triage_result', {}).get('domain_context', 'Unknown')}")
        
        # Extract Stage 2 metadata
        stage_2 = blueprint.get('stage_2_precision_discovery', {})
        if stage_2:
            datasets = stage_2.get('datasets', [])
            if datasets:
                insights.append("\nDATA SOURCES:")
                for dataset in datasets:
                    insights.append(f"- {dataset.get('table_name', 'Unknown')}: {dataset.get('business_purpose', 'No purpose defined')}")
                    insights.append(f"  Data Owner: {dataset.get('data_owner', 'Unknown')}")
                    insights.append(f"  Source System: {dataset.get('source_system', 'Unknown')}")
            
            # Add data quirks for quality context
            llm_hints = stage_2.get('llm_hints_discovered', {})
            data_quirks = llm_hints.get('data_quirks', [])
            if data_quirks:
                insights.append("\nDATA QUALITY CONTEXT:")
                for quirk in data_quirks:
                    insights.append(f"- {quirk}")
        
        return "\n".join(insights)
    
    def _get_fallback_analysis(self, question: str, results: Dict) -> Dict:
        """Fallback analysis when AI fails"""
        
        if results.get('status') == 'error':
            return {
                "direct_answer": "Query execution failed - unable to provide analysis",
                "key_insights": ["Query execution error prevented analysis"],
                "strategic_analysis": {
                    "business_impact": "Data access issues preventing business insights",
                    "confidence_level": "LOW",
                    "risk_factors": ["Data availability", "Query execution"],
                    "opportunities": ["Resolve technical issues"],
                    "strategic_recommendations": ["Fix query execution problems", "Verify data access"]
                },
                "metadata_enhanced_insights": {
                    "data_quality_assessment": "Unable to assess - query failed",
                    "business_context_application": "No business context available due to query failure",
                    "semantic_interpretation": "No semantic analysis possible",
                    "relationship_insights": "No relationship analysis possible"
                },
                "performance_insights": "Query execution failed"
            }
        
        rows = results.get('rows', [])
        return {
            "direct_answer": f"Analysis completed: {len(rows)} records retrieved",
            "key_insights": [f"Query returned {len(rows)} records successfully"],
            "strategic_analysis": {
                "business_impact": "Data retrieved successfully for analysis",
                "confidence_level": "MEDIUM",
                "risk_factors": ["Limited analysis due to AI unavailability"],
                "opportunities": ["Data available for manual analysis"],
                "strategic_recommendations": ["Review results manually", "Re-run analysis when AI available"]
            },
            "metadata_enhanced_insights": {
                "data_quality_assessment": f"Retrieved {len(rows)} records successfully",
                "business_context_application": "Basic fallback analysis applied",
                "semantic_interpretation": "Manual interpretation required",
                "relationship_insights": "Cross-table analysis opportunities available"
            },
            "performance_insights": "Query executed successfully"
        }


# Initialize engines
comprehensive_query_engine = ComprehensiveAIQueryEngine()
comprehensive_analysis_engine = ComprehensiveAnalysisEngine()


# ===== TOOL 1: COMPREHENSIVE SQL GENERATION =====
@tool
def sql_generation_tool(question: str, context_json: str) -> str:
    """
    COMPREHENSIVE AI-driven SQL generation using ALL Weaviate metadata
    
    Prevents hallucination by strictly constraining to discovered schema elements:
    - Uses exact table and column names from Weaviate
    - Applies semantic types for appropriate function selection  
    - Leverages sample values for format-aware filtering
    - Follows proven query patterns from answerable_questions
    - Handles data quirks and applies optimization hints
    - Uses relationship information for accurate JOINs
    
    Args:
        question: The business question to answer
        context_json: Complete adaptive blueprint with all metadata
    
    Returns:
        AI-generated SQL query using only discovered schema elements
    """
    logger.info("🧠 COMPREHENSIVE SQL Generation Tool started")
    
    try:
        # Parse the complete blueprint
        if isinstance(context_json, str):
            context = json.loads(context_json)
        else:
            context = context_json
        
        logger.info(f"📋 Question: {question}")
        logger.info(f"🎯 Blueprint keys: {list(context.keys())}")
        
        # Generate SQL using comprehensive metadata
        comprehensive_sql = comprehensive_query_engine.generate_comprehensive_sql(question, context)
        
        if not comprehensive_sql or "error" in comprehensive_sql.lower():
            logger.error("❌ Comprehensive SQL generation failed")
            return "SELECT 'Comprehensive SQL generation failed' as error_message"
        
        logger.info("✅ Comprehensive SQL generation completed successfully")
        logger.info(f"🎯 Generated SQL length: {len(comprehensive_sql)} characters")
        
        return comprehensive_sql
        
    except Exception as e:
        logger.error(f"❌ SQL generation tool error: {e}")
        return f"SELECT 'SQL generation error: {str(e)}' as error_message"


# ===== TOOL 2: ENHANCED ATHENA EXECUTION =====
@tool  
def athena_execution_tool(sql_query: str) -> str:
    """
    Enhanced Athena execution with comprehensive monitoring and validation
    
    Args:
        sql_query: The SQL query to execute
    
    Returns:
        JSON string with results, performance metrics, or error information
    """
    logger.info("🚀 Enhanced Athena Execution Tool started")
    
    # Comprehensive input validation
    if not sql_query or sql_query.strip() == "":
        return json.dumps({
            "status": "error",
            "error": {
                "code": "INVALID_INPUT",
                "message": "Empty or invalid SQL query provided"
            }
        }, indent=2)
    
    # Check for error messages in SQL
    if "error" in sql_query.lower() and "select" in sql_query.lower():
        return json.dumps({
            "status": "error", 
            "error": {
                "code": "SQL_GENERATION_ERROR",
                "message": "SQL generation failed - received error message instead of query",
                "sql_received": sql_query[:200]
            }
        }, indent=2)
    
    try:
        # Validate environment configuration
        required_env_vars = {
            'ATHENA_DATABASE': os.getenv('ATHENA_DATABASE'),
            'ATHENA_OUTPUT_LOCATION': os.getenv('ATHENA_OUTPUT_LOCATION')
        }
        
        missing_vars = [key for key, value in required_env_vars.items() if not value]
        if missing_vars:
            return json.dumps({
                "status": "error",
                "error": {
                    "code": "CONFIGURATION_ERROR", 
                    "message": f"Missing required environment variables: {missing_vars}",
                    "required_vars": list(required_env_vars.keys())
                }
            }, indent=2)
        
        # Initialize Athena client with validation
        try:
            athena_client = boto3.client(
                'athena',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            # Test client connectivity
            athena_client.list_work_groups(MaxResults=1)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": {
                    "code": "CLIENT_INITIALIZATION_ERROR",
                    "message": f"Failed to initialize Athena client: {str(e)}"
                }
            }, indent=2)
        
        # Execute query with comprehensive monitoring
        database = required_env_vars['ATHENA_DATABASE']
        output_location = required_env_vars['ATHENA_OUTPUT_LOCATION']
        workgroup = os.getenv('ATHENA_WORKGROUP', 'primary')
        
        logger.info(f"🎯 Executing in database: {database}")
        logger.info(f"📂 Output location: {output_location}")
        logger.info(f"🏢 Workgroup: {workgroup}")
        
        # Start execution with timeout monitoring
        start_time = time.time()
        
        try:
            start_response = athena_client.start_query_execution(
                QueryString=sql_query,
                ResultConfiguration={
                    'OutputLocation': output_location,
                    'EncryptionConfiguration': {
                        'EncryptionOption': 'SSE_S3'
                    }
                },
                QueryExecutionContext={
                    'Database': database
                },
                WorkGroup=workgroup
            )
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": {
                    "code": "QUERY_START_FAILED",
                    "message": f"Failed to start query execution: {str(e)}",
                    "sql_query": sql_query[:500]
                }
            }, indent=2)
        
        query_execution_id = start_response['QueryExecutionId']
        logger.info(f"🆔 Query ID: {query_execution_id}")
        
        # Poll for completion with progress monitoring
        max_attempts = 120  # 10 minutes max
        poll_interval = 2
        
        for attempt in range(max_attempts):
            try:
                response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
                execution = response['QueryExecution']
                status = execution['Status']['State']
                
                if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                
                # Log progress periodically
                if attempt % 15 == 0 and attempt > 0:  # Every 30 seconds
                    elapsed = attempt * poll_interval
                    logger.info(f"⏳ Query still running... ({elapsed}s elapsed)")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.warning(f"⚠️ Error polling query status: {e}")
                time.sleep(poll_interval)
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        if status == 'SUCCEEDED':
            try:
                # Get comprehensive results
                results_response = athena_client.get_query_results(
                    QueryExecutionId=query_execution_id,
                    MaxResults=1000
                )
                
                # Process results with error handling
                result_set = results_response['ResultSet']
                column_info = result_set['ResultSetMetadata']['ColumnInfo']
                columns = [col['Label'] for col in column_info]
                column_types = [col['Type'] for col in column_info]
                
                rows = []
                result_rows = result_set['Rows']
                data_rows = result_rows[1:] if len(result_rows) > 1 else []
                
                for row in data_rows:
                    row_data = []
                    for cell in row['Data']:
                        value = cell.get('VarCharValue', '')
                        # Convert empty strings to None for better data handling
                        row_data.append(value if value != '' else None)
                    rows.append(row_data)
                
                # Extract comprehensive performance statistics
                statistics = execution.get('Statistics', {})
                data_scanned_bytes = statistics.get('DataScannedInBytes', 0)
                data_scanned_mb = round(data_scanned_bytes / (1024 * 1024), 2)
                
                # Calculate estimated cost (approximate)
                cost_per_tb = 5.0  # $5 per TB scanned
                estimated_cost = round((data_scanned_bytes / (1024**4)) * cost_per_tb, 6)
                
                response_data = {
                    "status": "success",
                    "columns": columns,
                    "column_types": column_types,
                    "rows": rows,
                    "row_count": len(rows),
                    "performance_metrics": {
                        "execution_time_ms": execution_time_ms,
                        "engine_execution_time_ms": statistics.get('EngineExecutionTimeInMillis', execution_time_ms),
                        "data_scanned_mb": data_scanned_mb,
                        "data_scanned_bytes": data_scanned_bytes,
                        "query_queue_time_ms": statistics.get('QueryQueueTimeInMillis', 0),
                        "query_planning_time_ms": statistics.get('QueryPlanningTimeInMillis', 0),
                        "estimated_cost_usd": estimated_cost
                    },
                    "query_metadata": {
                        "query_id": query_execution_id,
                        "database": database,
                        "workgroup": workgroup,
                        "output_location": output_location
                    }
                }
                
                logger.info(f"✅ Query executed successfully: {len(rows)} rows, {data_scanned_mb} MB scanned")
                
            except Exception as e:
                logger.error(f"❌ Error processing query results: {e}")
                response_data = {
                    "status": "error",
                    "error": {
                        "code": "RESULT_PROCESSING_ERROR",
                        "message": f"Query succeeded but failed to process results: {str(e)}",
                        "query_id": query_execution_id,
                        "execution_time_ms": execution_time_ms
                    }
                }
        
        else:
            # Query execution failed
            error_reason = execution['Status'].get('StateChangeReason', 'Query execution failed')
            failure_reason = execution['Status'].get('AthenaError', {})
            
            response_data = {
                "status": "error",
                "error": {
                    "code": "QUERY_EXECUTION_FAILED",
                    "message": error_reason,
                    "athena_error": failure_reason,
                    "query_id": query_execution_id,
                    "execution_time_ms": execution_time_ms,
                    "sql_query": sql_query[:500]  # Include SQL for debugging
                }
            }
            
            logger.error(f"❌ Query execution failed: {error_reason}")
        
        return json.dumps(response_data, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Athena execution error: {e}")
        return json.dumps({
            "status": "error",
            "error": {
                "code": "EXECUTION_ERROR",
                "message": str(e),
                "type": type(e).__name__
            }
        }, indent=2)


# ===== TOOL 3: COMPREHENSIVE DATA ANALYSIS =====
@tool
def data_analysis_tool(question: str, athena_results_json: str, original_blueprint_json: str = None) -> str:
    """
    COMPREHENSIVE AI-driven data analysis using blueprint metadata for enhanced insights
    
    Args:
        question: The original business question
        athena_results_json: JSON string with Athena execution results
        original_blueprint_json: Optional original blueprint for metadata context
    
    Returns:
        JSON string with comprehensive AI-generated analysis and strategic insights
    """
    logger.info("🧠 COMPREHENSIVE Data Analysis Tool started")
    
    try:
        # Parse Athena results
        if isinstance(athena_results_json, str):
            results = json.loads(athena_results_json)
        else:
            results = athena_results_json
        
        # Parse original blueprint if provided
        original_blueprint = None
        if original_blueprint_json:
            try:
                if isinstance(original_blueprint_json, str):
                    original_blueprint = json.loads(original_blueprint_json)
                else:
                    original_blueprint = original_blueprint_json
            except:
                logger.warning("⚠️ Failed to parse original blueprint, continuing without metadata context")
        
        logger.info(f"🧠 Analyzing results for: {question}")
        logger.info(f"📊 Results status: {results.get('status', 'unknown')}")
        logger.info(f"🎯 Blueprint context: {'Available' if original_blueprint else 'Not available'}")
        
        # Generate comprehensive analysis using metadata
        comprehensive_analysis = comprehensive_analysis_engine.analyze_results_with_metadata(
            question, results, original_blueprint
        )
        
        logger.info("✅ Comprehensive analysis completed successfully")
        return json.dumps(comprehensive_analysis, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Data analysis error: {e}")
        return json.dumps({
            "direct_answer": f"Analysis error: {str(e)}",
            "key_insights": ["Comprehensive analysis could not be completed due to technical error"],
            "strategic_analysis": {
                "business_impact": "Unable to provide insights due to analysis error",
                "confidence_level": "LOW",
                "risk_factors": ["Technical analysis issues", "Error in data processing"],
                "opportunities": ["Resolve technical issues", "Retry analysis"],
                "strategic_recommendations": ["Check data format", "Verify analysis requirements", "Review error logs"]
            },
            "metadata_enhanced_insights": {
                "data_quality_assessment": "Analysis failed due to technical error",
                "business_context_application": "No business context analysis performed",
                "semantic_interpretation": "No semantic analysis possible due to error",
                "relationship_insights": "No relationship analysis performed"
            },
            "performance_insights": "Unable to assess performance due to analysis failure",
            "error_details": str(e)
        }, indent=2)