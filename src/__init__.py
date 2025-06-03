"""
Auto-Analyst Source Package
---------------------------

This package contains the core logic for the Auto-Analyst application,
including the agent definitions, data retrieval mechanisms, and the
enhanced analysis capabilities.
"""

# Core agent systems
from .agents import auto_analyst, auto_analyst_ind

# Memory management agents
from .memory_agents import memory_summarize_agent, error_memory_agent

# Data retrieval and preparation
from .retrievers import make_data, initiatlize_retrievers, styling_instructions, correct_num, return_vals

# Enhanced Agents Coordinator
# Assuming AgentCoordinator will be defined in enhanced_agents.coordinator
# If it's integrated into agents.py, this import might change or be removed.
try:
    from .enhanced_agents.coordinator import AgentCoordinator
except ImportError:
    # Fallback if coordinator is part of the main agents.py or not yet created
    # This allows the package to be partially used even if some modules are pending
    AgentCoordinator = None 
    print("Warning: AgentCoordinator not found. Enhanced agent orchestration might be limited.")


# Individual Enhanced Agents
# These can be directly accessed or orchestrated via AgentCoordinator
try:
    from .enhanced_agents.statistical import EnhancedStatisticalAgent
except ImportError:
    EnhancedStatisticalAgent = None
    print("Warning: EnhancedStatisticalAgent not found.")

try:
    from .enhanced_agents.ml import AdvancedMLAgent
except ImportError:
    AdvancedMLAgent = None
    print("Warning: AdvancedMLAgent not found.")

try:
    from .enhanced_agents.kpi import KPIGeneratorAgent
except ImportError:
    KPIGeneratorAgent = None
    print("Warning: KPIGeneratorAgent not found.")

try:
    from .enhanced_agents.quality import DataQualityAgent
except ImportError:
    DataQualityAgent = None
    print("Warning: DataQualityAgent not found.")

try:
    from .enhanced_agents.visualization import AdvancedVisualizationAgent
except ImportError:
    AdvancedVisualizationAgent = None
    print("Warning: AdvancedVisualizationAgent not found.")

try:
    from .enhanced_agents.excel_integration import ExcelIntegrationAgent
except ImportError:
    ExcelIntegrationAgent = None
    print("Warning: ExcelIntegrationAgent not found.")


__all__ = [
    # Core systems
    'auto_analyst',
    'auto_analyst_ind',
    # Memory agents
    'memory_summarize_agent',
    'error_memory_agent',
    # Retrievers
    'make_data',
    'initiatlize_retrievers',
    'styling_instructions',
    'correct_num',
    'return_vals',
    # Enhanced agents & coordinator
    'AgentCoordinator',
    'EnhancedStatisticalAgent',
    'AdvancedMLAgent',
    'KPIGeneratorAgent',
    'DataQualityAgent',
    'AdvancedVisualizationAgent',
    'ExcelIntegrationAgent',
]

# Print a message upon package import for clarity during development
print("Auto-Analyst 'src' package initialized.")
if AgentCoordinator is None:
    print("Note: Some enhanced features might be unavailable until all modules are created.")
