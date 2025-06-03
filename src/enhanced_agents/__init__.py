"""
Enhanced Agents Package for Auto-Analyst
-----------------------------------------

This package groups all the advanced AI agents designed to provide
more sophisticated data analysis capabilities. Each agent specializes
in a specific domain, and they can be orchestrated by the AgentCoordinator.
"""

# Import individual enhanced agents
try:
    from .statistical import EnhancedStatisticalAgent
except ImportError:
    EnhancedStatisticalAgent = None
    print("Warning: EnhancedStatisticalAgent module not found or class not defined within.")

try:
    from .ml import AdvancedMLAgent
except ImportError:
    AdvancedMLAgent = None
    print("Warning: AdvancedMLAgent module not found or class not defined within.")

try:
    from .kpi import KPIGeneratorAgent
except ImportError:
    KPIGeneratorAgent = None
    print("Warning: KPIGeneratorAgent module not found or class not defined within.")

try:
    from .quality import DataQualityAgent
except ImportError:
    DataQualityAgent = None
    print("Warning: DataQualityAgent module not found or class not defined within.")

try:
    from .visualization import AdvancedVisualizationAgent
except ImportError:
    AdvancedVisualizationAgent = None
    print("Warning: AdvancedVisualizationAgent module not found or class not defined within.")

try:
    from .excel_integration import ExcelIntegrationAgent
except ImportError:
    ExcelIntegrationAgent = None
    print("Warning: ExcelIntegrationAgent module not found or class not defined within.")

# Import the Agent Coordinator
try:
    from .coordinator import AgentCoordinator
except ImportError:
    AgentCoordinator = None
    print("Warning: AgentCoordinator module not found or class not defined within. Multi-agent orchestration will be affected.")

# Define what is available for direct import from this package
__all__ = [
    'EnhancedStatisticalAgent',
    'AdvancedMLAgent',
    'KPIGeneratorAgent',
    'DataQualityAgent',
    'AdvancedVisualizationAgent',
    'ExcelIntegrationAgent',
    'AgentCoordinator',
]

# Optional: A message to confirm package initialization
# print("Auto-Analyst 'enhanced_agents' package initialized.")
# if None in __all__: # A bit of a hacky check, but indicates some modules failed to load
#     missing_modules = [name for name, cls in zip(__all__, [EnhancedStatisticalAgent, AdvancedMLAgent, KPIGeneratorAgent, DataQualityAgent, AdvancedVisualizationAgent, ExcelIntegrationAgent, AgentCoordinator]) if cls is None]
#     if missing_modules:
#         print(f"Note: The following enhanced agent modules are not fully loaded: {', '.join(missing_modules)}")

