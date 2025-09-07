# src/iML/agents/tutorial_retriever_agent.py

import logging
from typing import List, Dict, Any, Optional

from .base_agent import BaseAgent
from ..utils.tutorial_retriever import TutorialRetriever, TutorialInfo

logger = logging.getLogger(__name__)


class TutorialRetrieverAgent(BaseAgent):
    """
    Agent for retrieving relevant tutorials based on iML's analysis results.
    Similar to autogluon-assistant's RetrieverAgent but adapted for iML's specific features.
    """
    
    def __init__(self, config, manager, tutorial_base_path: str = "src/iML/tutorial"):
        super().__init__(config=config, manager=manager)
        self.tutorial_retriever = TutorialRetriever(tutorial_base_path)
        self.retrieved_tutorials: List[TutorialInfo] = []
    
    def __call__(self, max_tutorials: int = 2) -> List[TutorialInfo]:
        """
        Retrieve relevant tutorials based on current analysis results.
        
        Args:
            max_tutorials: Maximum number of tutorials to retrieve
            
        Returns:
            List of relevant TutorialInfo objects
        """
        self.manager.log_agent_start("TutorialRetrieverAgent: Retrieving relevant tutorials...")
        
        try:
            # Get analysis results from manager
            description_analysis = getattr(self.manager, 'description_analysis', None)
            profiling_summary = getattr(self.manager, 'profiling_summary', None)
            
            if not description_analysis:
                logger.warning("No description analysis available for tutorial retrieval")
                self._log_no_tutorials("No description analysis available")
                return []
            
            # Retrieve tutorials
            self.retrieved_tutorials = self.tutorial_retriever.retrieve_tutorials(
                description_analysis=description_analysis,
                profiling_summary=profiling_summary,
                max_tutorials=max_tutorials
            )
            
            if self.retrieved_tutorials:
                tutorial_names = [t.title for t in self.retrieved_tutorials]
                logger.info(f"Retrieved {len(self.retrieved_tutorials)} relevant tutorials: {tutorial_names}")
                self._log_tutorial_usage(self.retrieved_tutorials)
            else:
                logger.info("No relevant tutorials found")
                self._log_no_tutorials("No matching tutorials found")
            
            # Save results for debugging
            self._save_retrieval_results()
            
            self.manager.log_agent_end(f"TutorialRetrieverAgent: Retrieved {len(self.retrieved_tutorials)} tutorials")
            
            return self.retrieved_tutorials
            
        except Exception as e:
            logger.error(f"Error during tutorial retrieval: {e}")
            self._log_no_tutorials(f"Error during retrieval: {e}")
            return []
    
    def _log_tutorial_usage(self, tutorials: List[TutorialInfo]) -> None:
        """Log which tutorials are being used."""
        tutorial_list = []
        for tutorial in tutorials:
            relative_path = tutorial.path.relative_to(tutorial.path.parents[2])  # relative to src/iML/
            tutorial_list.append(f"{tutorial.title} (score: {tutorial.score:.2f}) - {relative_path}")
        
        log_message = f"Using tutorials:\n" + "\n".join(f"  - {t}" for t in tutorial_list)
        logger.info(log_message)
        print(log_message)  # Also print to console for user visibility
    
    def _log_no_tutorials(self, reason: str) -> None:
        """Log when no tutorials are used."""
        log_message = f"No tutorials will be used. Reason: {reason}"
        logger.info(log_message)
        print(log_message)  # Also print to console for user visibility
    
    def _save_retrieval_results(self) -> None:
        """Save tutorial retrieval results for debugging."""
        if self.retrieved_tutorials:
            formatted_results = self.tutorial_retriever.format_tutorials_for_prompt(self.retrieved_tutorials)
            self.manager.save_and_log_states(
                content=formatted_results,
                save_name="tutorial_retrieval_results.txt",
                add_uuid=False
            )
        else:
            self.manager.save_and_log_states(
                content="No tutorials retrieved",
                save_name="tutorial_retrieval_results.txt",
                add_uuid=False
            )
    
    def get_tutorials_for_prompt(self) -> str:
        """Get formatted tutorials for inclusion in prompts."""
        if not self.retrieved_tutorials:
            return ""
        
        return self.tutorial_retriever.format_tutorials_for_prompt(self.retrieved_tutorials)
