# src/iML/utils/tutorial_retriever.py

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import difflib

logger = logging.getLogger(__name__)


class TutorialInfo:
    """Stores information about a tutorial"""
    def __init__(self, path: Path, title: str, content: str, score: float = 0.0):
        self.path = path
        self.title = title
        self.content = content
        self.score = score


class TutorialRetriever:
    """
    Retrieves relevant tutorials based on iML's analysis results.
    Uses simple keyword-based matching inspired by autogluon-assistant's approach.
    """
    
    def __init__(self, tutorial_base_path: str = "src/iML/tutorial"):
        self.tutorial_base_path = Path(tutorial_base_path)
        self.tutorials_cache = None
        self._load_tutorials()
    
    def _load_tutorials(self) -> None:
        """Load all tutorials from the tutorial directory."""
        self.tutorials_cache = []
        
        if not self.tutorial_base_path.exists():
            logger.warning(f"Tutorial directory not found: {self.tutorial_base_path}")
            return
        
        # Recursively find all .md files
        for md_file in self.tutorial_base_path.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Extract title from first line or filename
                lines = content.split('\n')
                title = lines[0].strip().lstrip('#').strip() if lines else md_file.stem
                
                tutorial_info = TutorialInfo(
                    path=md_file,
                    title=title,
                    content=content
                )
                self.tutorials_cache.append(tutorial_info)
                
            except Exception as e:
                logger.warning(f"Error reading tutorial file {md_file}: {e}")
    
    def retrieve_tutorials(
        self,
        description_analysis: Dict[str, Any],
        profiling_summary: Optional[Dict[str, Any]] = None,
        max_tutorials: int = 2
    ) -> List[TutorialInfo]:
        """
        Retrieve relevant tutorials based on analysis results.
        
        Args:
            description_analysis: Results from DescriptionAnalyzerAgent
            profiling_summary: Results from ProfilingSummarizerAgent (optional)
            max_tutorials: Maximum number of tutorials to return
            
        Returns:
            List of relevant TutorialInfo objects
        """
        if not self.tutorials_cache:
            logger.info("No tutorials available in cache")
            return []
        
        # Generate search keywords based on iML's analysis
        keywords = self._generate_keywords(description_analysis, profiling_summary)
        
        if not keywords:
            logger.info("No keywords generated for tutorial search")
            return []
        
        # Score tutorials based on keyword matching
        scored_tutorials = []
        for tutorial in self.tutorials_cache:
            score = self._calculate_relevance_score(tutorial, keywords)
            if score > 0:
                tutorial.score = score
                scored_tutorials.append(tutorial)
        
        # Sort by score and return top results
        scored_tutorials.sort(key=lambda t: t.score, reverse=True)
        selected_tutorials = scored_tutorials[:max_tutorials]
        
        return selected_tutorials
    
    def _generate_keywords(
        self,
        description_analysis: Dict[str, Any],
        profiling_summary: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate search keywords based on iML's analysis results."""
        keywords = []
        
        # Keywords from task type
        task_type = description_analysis.get('task_type', '').lower()
        if task_type:
            keywords.append(task_type)
            
            # Map task types to domain keywords
            if 'text' in task_type or 'nlp' in task_type:
                keywords.extend(['text', 'nlp', 'bert', 'tokenization', 'preprocessing'])
            elif 'image' in task_type or 'vision' in task_type:
                keywords.extend(['image', 'computer vision', 'cnn', 'augmentation'])
            elif 'tabular' in task_type:
                keywords.extend(['tabular', 'structured data', 'pandas'])
                
                # Check for imbalanced data from profiling summary
                if profiling_summary:
                    label_analysis = profiling_summary.get('label_analysis', {})
                    imbalance = label_analysis.get('class_distribution_imbalance', '')
                    if imbalance and imbalance != 'none':
                        keywords.extend(['imbalanced', 'class imbalance', 'sampling'])
        
        # Keywords from task description
        task_desc = description_analysis.get('task', '').lower()
        if 'classification' in task_desc:
            keywords.append('classification')
        elif 'regression' in task_desc:
            keywords.append('regression')
        
        # Keywords from input/output data description
        input_data = description_analysis.get('input_data', '').lower()
        if 'image' in input_data:
            keywords.extend(['image', 'jpg', 'png'])
        elif 'text' in input_data:
            keywords.extend(['text', 'natural language'])
        
        # Keywords from profiling summary (iML specific features)
        if profiling_summary:
            feature_quality = profiling_summary.get('feature_quality', {})
            
            # High missing data
            high_missing = feature_quality.get('high_missing_columns', [])
            if high_missing:
                keywords.extend(['missing values', 'imputation'])
            
            # High cardinality categoricals
            high_card = feature_quality.get('high_cardinality_categoricals', [])
            if high_card:
                keywords.extend(['categorical', 'encoding', 'cardinality'])
            
            # Date columns
            date_cols = feature_quality.get('date_like_cols', [])
            if date_cols:
                keywords.extend(['date', 'time series', 'temporal'])
        
        # Remove duplicates and return
        return list(set(keywords))
    
    def _calculate_relevance_score(self, tutorial: TutorialInfo, keywords: List[str]) -> float:
        """Calculate relevance score between tutorial and keywords."""
        if not keywords:
            return 0.0
        
        # Combine title and content for searching (weight title more heavily)
        title_lower = tutorial.title.lower()
        content_lower = tutorial.content.lower()
        
        score = 0.0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Title matches get higher weight
            if keyword_lower in title_lower:
                score += 3.0
            
            # Content matches
            content_matches = content_lower.count(keyword_lower)
            score += content_matches * 0.1
            
            # Path-based matching (e.g., tabular/image/text directories)
            path_str = str(tutorial.path).lower()
            if keyword_lower in path_str:
                score += 2.0
        
        # Normalize by number of keywords to prevent bias toward longer keyword lists
        return score / len(keywords) if keywords else 0.0
    
    def format_tutorials_for_prompt(self, tutorials: List[TutorialInfo]) -> str:
        """Format selected tutorials for inclusion in prompts."""
        if not tutorials:
            return ""
        
        formatted_sections = []
        for i, tutorial in enumerate(tutorials, 1):
            section = f"## Tutorial {i}: {tutorial.title}\n"
            section += f"**Path:** {tutorial.path.relative_to(self.tutorial_base_path)}\n"
            section += f"**Relevance Score:** {tutorial.score:.2f}\n\n"
            
            # Include first 500 characters of content as preview
            preview = tutorial.content[:500].strip()
            if len(tutorial.content) > 500:
                preview += "..."
            section += f"**Content Preview:**\n{preview}\n\n"
            
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)
