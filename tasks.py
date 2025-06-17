# tasks.py - UPDATED FOR ENHANCED QUERY ANALYST PIPELINE
import os
import yaml
from pathlib import Path
from crewai import Task
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveForecastingTasks:
    """
    Enhanced Adaptive Forecasting Tasks class supporting the NEW 5-stage pipeline:
    0. Query Analysis (NEW)
    1. Schema Discovery (Updated)
    2. SQL Generation 
    3. Athena Execution
    4. Data Analysis
    """
    
    def __init__(self):
        # Load adaptive task configurations from YAML
        config_path = Path(__file__).parent / "config" / "tasks.yaml"
        try:
            with open(config_path, 'r') as file:
                self.tasks_config = yaml.safe_load(file)
                logger.info("✅ Enhanced task configurations loaded successfully")
        except FileNotFoundError:
            raise FileNotFoundError(f"Adaptive tasks.yaml not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing adaptive tasks.yaml: {e}")
        
        # Validate that all required tasks are present
        required_tasks = [
            'query_analysis_task',  # NEW TASK
            'adaptive_context_task', 
            'adaptive_sql_generation_task', 
            'adaptive_execution_task', 
            'adaptive_data_analysis_task'
        ]
        missing_tasks = [task for task in required_tasks if task not in self.tasks_config]
        if missing_tasks:
            logger.warning(f"Missing tasks in tasks.yaml: {missing_tasks} - will create fallback definitions")
            
        logger.info(f"📋 Enhanced tasks available: {list(self.tasks_config.keys())}")
    
    def _create_task(self, task_name: str, agent, context: Optional[List[Task]] = None, **kwargs) -> Task:
        """
        Create an enhanced task from YAML configuration with proper context handling.
        """
        config = self.tasks_config.get(task_name)
        if not config:
            # Create fallback task if not in YAML
            logger.warning(f"Task '{task_name}' not found in YAML, creating fallback")
            return self._create_fallback_task(task_name, agent, context, **kwargs)
        
        # Ensure required fields exist
        required_fields = ['description', 'expected_output']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' for task '{task_name}'")
        
        # Replace placeholders in description and expected_output
        description = config['description']
        expected_output = config['expected_output']
        
        # Replace {question} placeholder if provided
        if 'question' in kwargs:
            description = description.replace('{question}', kwargs['question'])
            expected_output = expected_output.replace('{question}', kwargs['question'])
            logger.debug(f"🔄 Replaced question placeholder in {task_name}")
        
        # Create enhanced task with proper parameters
        task_params = {
            'description': description,
            'expected_output': expected_output,
            'agent': agent
        }
        
        # Add context chain if provided (enables multi-task intelligence flow)
        if context:
            task_params['context'] = context
            logger.debug(f"🔗 Added context chain to {task_name}: {len(context)} predecessor tasks")
        
        logger.info(f"📋 Created enhanced {task_name} with {'context chain' if context else 'no dependencies'}")
        return Task(**task_params)
    
    def _create_fallback_task(self, task_name: str, agent, context: Optional[List[Task]] = None, **kwargs) -> Task:
        """Create fallback task when not found in YAML"""
        
        fallback_descriptions = {
            'query_analysis_task': f"Analyze the user's question: '{kwargs.get('question', 'Unknown')}'. Use the Query Analyzer Tool to decompose it into a structured JSON plan.",
            'adaptive_context_task': "Using the JSON plan from the Query Analysis Task, discover the complete database schema using the Adaptive Schema Discovery Engine.",
            'adaptive_sql_generation_task': f"Generate SQL for: '{kwargs.get('question', 'Unknown')}' using the discovered schema blueprint.",
            'adaptive_execution_task': "Execute the SQL query using Athena Query Execution Tool.",
            'adaptive_data_analysis_task': f"Analyze results for: '{kwargs.get('question', 'Unknown')}' using Enhanced Data Analysis and Forecasting Tool."
        }
        
        fallback_outputs = {
            'query_analysis_task': "A JSON object containing the decomposed plan with 'primary_intent', 'key_entities', 'metrics_to_calculate', and 'sub_questions_for_schema_search'.",
            'adaptive_context_task': "A comprehensive adaptive blueprint with discovered schema details, ready for SQL generation.",
            'adaptive_sql_generation_task': "A perfect SQL query using discovered schema elements.",
            'adaptive_execution_task': "Query execution results with performance metrics.",
            'adaptive_data_analysis_task': "Natural language business analysis with answer, reasoning, and recommendations."
        }
        
        task_params = {
            'description': fallback_descriptions.get(task_name, f"Execute {task_name}"),
            'expected_output': fallback_outputs.get(task_name, f"Output from {task_name}"),
            'agent': agent
        }
        
        if context:
            task_params['context'] = context
        
        logger.info(f"🔄 Created fallback {task_name}")
        return Task(**task_params)
    
    def query_analysis_task(self, agent, question: str) -> Task:
        """
        🧠 STAGE 0: NEW - Create the intelligent query analysis task
        
        This task enables the Query Analyst to:
        1. Decompose complex business questions into structured components
        2. Identify primary business intent and key entities
        3. Create targeted search queries for schema discovery
        4. Predict needed data types for optimization
        5. Assess query complexity for adaptive processing
        """
        logger.info(f"🧠 Creating query analysis task for: '{question}'")
        logger.info(f"   Agent tools: Query Analyzer Tool")
        logger.info(f"   AI Features: Question decomposition, intent analysis, search optimization")
        
        return self._create_task('query_analysis_task', agent, question=question)
    
    def adaptive_context_task(self, agent, question: str, context_task: Task) -> Task:
        """
        🔍 STAGE 1: Updated adaptive context analysis task
        
        This task now receives structured plans from Query Analyst and:
        1. Uses "sub_questions_for_schema_search" for targeted discovery
        2. Applies precision schema discovery with AI scoring
        3. Generates comprehensive blueprints for SQL generation
        """
        logger.info(f"🔍 Creating enhanced adaptive context task")
        logger.info(f"   Question: '{question}'")
        logger.info(f"   Agent tools: Adaptive Schema Discovery Engine")
        logger.info(f"   Context dependency: Structured query plan from Query Analyst")
        
        return self._create_task(
            'adaptive_context_task', 
            agent, 
            context=[context_task],
            question=question
        )
    
    def adaptive_sql_generation_task(self, agent, question: str, context_task: Task) -> Task:
        """⚙️ STAGE 2: SQL Generation (unchanged logic, updated context)"""
        logger.info(f"⚙️ Creating adaptive SQL generation task")
        logger.info(f"   Context dependency: Enhanced blueprint from Query Analyst pipeline")
        
        return self._create_task(
            'adaptive_sql_generation_task', 
            agent, 
            context=[context_task],
            question=question
        )
    
    def adaptive_execution_task(self, agent, sql_generation_task: Task) -> Task:
        """🚀 STAGE 3: Athena Execution (unchanged)"""
        logger.info(f"🚀 Creating adaptive Athena execution task")
        
        return self._create_task(
            'adaptive_execution_task',
            agent,
            context=[sql_generation_task]
        )
    
    def adaptive_data_analysis_task(self, agent, question: str, execution_task: Task) -> Task:
        """📊 STAGE 4: Data Analysis (unchanged)"""
        logger.info(f"📊 Creating adaptive data analysis task")
        
        return self._create_task(
            'adaptive_data_analysis_task',
            agent,
            context=[execution_task],
            question=question
        )
    
    def create_enhanced_adaptive_pipeline(self, agents_instance, question: str) -> List[Task]:
        """
        🎯 Create the complete ENHANCED 5-stage adaptive pipeline
        
        NEW PIPELINE FLOW:
        Stage 0: Query Analysis → Stage 1: Schema Discovery → Stage 2: SQL Generation → Stage 3: Execution → Stage 4: Analysis
        
        Args:
            agents_instance: AdaptiveForecastingAgents instance
            question: User's natural language question
            
        Returns:
            List[Task]: Complete enhanced pipeline with proper dependencies
        """
        logger.info(f"🎯 Creating ENHANCED 5-stage adaptive pipeline")
        logger.info(f"   Question: '{question}'")
        logger.info(f"   NEW Pipeline: Query Analysis → Schema Discovery → SQL Generation → Execution → Analysis")
        
        # Create enhanced agents (including new query_analyst)
        query_analyst = agents_instance.query_analyst()
        adaptive_context_analyst = agents_instance.adaptive_context_analyst()
        sql_developer = agents_instance.sql_developer()
        athena_executor = agents_instance.athena_executor()
        data_insights_analyst = agents_instance.data_insights_analyst()
        
        # Create enhanced task chain with NEW query analysis first
        analysis_task = self.query_analysis_task(query_analyst, question)
        context_task = self.adaptive_context_task(adaptive_context_analyst, question, analysis_task)
        sql_task = self.adaptive_sql_generation_task(sql_developer, question, context_task)
        execution_task = self.adaptive_execution_task(athena_executor, sql_task)
        insights_task = self.adaptive_data_analysis_task(data_insights_analyst, question, execution_task)
        
        pipeline = [analysis_task, context_task, sql_task, execution_task, insights_task]
        
        logger.info(f"✅ Enhanced 5-stage pipeline created successfully")
        logger.info(f"   Tasks: {len(pipeline)} with intelligent dependency chain")
        logger.info(f"   Flow: Query Analysis → Schema Discovery → SQL Generation → Execution → Insights")
        
        return pipeline
    
    def validate_enhanced_pipeline(self, tasks: List[Task]) -> bool:
        """Validate that the enhanced task pipeline is properly configured."""
        if len(tasks) != 5:  # Updated for 5 stages
            raise ValueError(f"Enhanced pipeline must have exactly 5 tasks, got {len(tasks)}")
        
        # Validate task dependencies for enhanced pipeline
        expected_dependencies = [0, 1, 1, 1, 1]  # analysis:0, context:1, sql:1, execution:1, insights:1
        for i, (task, expected_deps) in enumerate(zip(tasks, expected_dependencies)):
            actual_deps = len(task.context) if hasattr(task, 'context') and task.context else 0
            if actual_deps != expected_deps:
                raise ValueError(f"Enhanced task {i} has {actual_deps} dependencies, expected {expected_deps}")
        
        logger.info("✅ Enhanced pipeline validation passed")
        return True

# Keep the original class for backward compatibility
class ForecastingTasks(AdaptiveForecastingTasks):
    """Backward compatibility class - inherits all enhanced functionality"""
    
    def create_adaptive_pipeline(self, agents_instance, question: str) -> List[Task]:
        """Backward compatibility method - calls enhanced pipeline"""
        return self.create_enhanced_adaptive_pipeline(agents_instance, question)