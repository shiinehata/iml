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
        self.is_debug_mode = False  # Track if we're in debug mode
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
        max_tutorials: int = 2,
        is_debug_mode: bool = False
    ) -> List[TutorialInfo]:
        """
        Retrieve relevant tutorials based on analysis results.
        
        Args:
            description_analysis: Results from DescriptionAnalyzerAgent
            profiling_summary: Results from ProfilingSummarizerAgent (optional)
            max_tutorials: Maximum number of tutorials to return
            is_debug_mode: Whether this is for debugging/fixing errors (no tutorials needed)
            
        Returns:
            List of relevant TutorialInfo objects
        """
        # Store debug mode state
        self.is_debug_mode = is_debug_mode
        
        # If in debug mode, don't return any tutorials
        if is_debug_mode:
            logger.info("Debug mode detected - skipping tutorial retrieval")
            return []
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
            
            # Map task types to domain keywords with specific library preferences
            if 'text' in task_type or 'nlp' in task_type:
                keywords.extend([
                    'text', 'nlp', 'bert', 'tokenization', 'preprocessing',
                    'tensorflow', 'keras', 'tensorflow_hub', 'transformers',
                    'sentiment', 'imdb', 'text_classification'
                ])
            elif 'image' in task_type or 'vision' in task_type:
                keywords.extend([
                    'image', 'computer vision', 'cnn', 'augmentation',
                    'tensorflow', 'keras', 'mobilenet', 'transfer learning',
                    'image_classification', 'cats_dogs', 'pretrained'
                ])
            elif 'tabular' in task_type:
                keywords.extend([
                    'tabular', 'structured data', 'pandas', 'csv',
                    'tensorflow', 'keras', 'preprocessing layers',
                    'normalization', 'categorical encoding', 'classification'
                ])
                
                # Check for imbalanced data from profiling summary
                if profiling_summary:
                    label_analysis = profiling_summary.get('label_analysis', {})
                    imbalance = label_analysis.get('class_distribution_imbalance', '')
                    if imbalance and imbalance != 'none':
                        keywords.extend(['imbalanced', 'class imbalance', 'sampling'])
        
        # Keywords from task description
        task_desc = description_analysis.get('task', '').lower()
        if 'classification' in task_desc:
            keywords.extend(['classification', 'binary_classification'])
        elif 'regression' in task_desc:
            keywords.append('regression')
        
        # Keywords from input/output data description
        input_data = description_analysis.get('input_data', '').lower()
        if 'image' in input_data:
            keywords.extend(['image', 'jpg', 'png', 'image_dataset_from_directory'])
        elif 'text' in input_data:
            keywords.extend(['text', 'natural language', 'text_dataset_from_directory'])
        
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
                keywords.extend(['categorical', 'encoding', 'cardinality', 'stringlookup'])
            
            # Date columns
            date_cols = feature_quality.get('date_like_cols', [])
            if date_cols:
                keywords.extend(['date', 'time series', 'temporal'])
        
        # Always prioritize TensorFlow/Keras ecosystem
        keywords.extend(['tensorflow', 'keras', 'tf.data'])
        
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
        
        # Priority libraries/frameworks that should be heavily weighted
        priority_libs = {
            'tensorflow': 5.0,
            'keras': 5.0,
            'tf.data': 4.0,
            'tf.keras': 5.0,
            'tensorflow_hub': 4.0,
            'mobilenet': 3.0,
            'bert': 4.0,
            'preprocessing layers': 3.0,
            'stringlookup': 3.0,
            'categoricalencoding': 3.0,
            'normalization': 3.0
        }
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Check if this is a priority library/framework
            weight_multiplier = 1.0
            for priority_lib, multiplier in priority_libs.items():
                if priority_lib in keyword_lower:
                    weight_multiplier = multiplier
                    break
            
            # Title matches get higher weight
            if keyword_lower in title_lower:
                score += 3.0 * weight_multiplier
            
            # Content matches with priority weighting
            content_matches = content_lower.count(keyword_lower)
            score += content_matches * 0.1 * weight_multiplier
            
            # Path-based matching (e.g., tabular/image/text directories)
            path_str = str(tutorial.path).lower()
            if keyword_lower in path_str:
                score += 2.0 * weight_multiplier
        
        # Bonus for tutorials that use recommended TensorFlow/Keras ecosystem
        tf_keras_bonus = 0.0
        if any(lib in content_lower for lib in ['tensorflow', 'keras', 'tf.keras']):
            tf_keras_bonus += 10.0
        
        if any(lib in content_lower for lib in ['tf.data', 'tensorflow_hub', 'hub.keraslayer']):
            tf_keras_bonus += 5.0
            
        if any(lib in content_lower for lib in ['preprocessing layers', 'stringlookup', 'categoricalencoding']):
            tf_keras_bonus += 3.0
        
        # Normalize by number of keywords and add bonus
        normalized_score = score / len(keywords) if keywords else 0.0
        return normalized_score + tf_keras_bonus
    
    def format_tutorials_for_prompt(self, tutorials: List[TutorialInfo]) -> str:
        """Format selected tutorials for inclusion in prompts."""
        if not tutorials:
            return ""
        
        # If in debug mode, return empty string (no tutorials needed for debugging)
        if self.is_debug_mode:
            return ""
        
        formatted_sections = []
        
        # Add header emphasizing the use of tutorial libraries
        header = """## PRIORITY: Use Libraries and Methods from These Tutorials

The following tutorials demonstrate the PREFERRED libraries and approaches for your task type. 
You should prioritize using the same libraries, methods, and patterns shown in these tutorials.

**COMPLETE TUTORIAL CONTENT INCLUDED BELOW - USE AS PRIMARY REFERENCE**

"""
        formatted_sections.append(header)
        
        for i, tutorial in enumerate(tutorials, 1):
            section = f"### Tutorial {i}: {tutorial.title}\n"
            section += f"**Path:** {tutorial.path.relative_to(self.tutorial_base_path)}\n"
            section += f"**Relevance Score:** {tutorial.score:.2f}\n\n"
            
            # Extract key libraries used in the tutorial
            content_lower = tutorial.content.lower()
            libraries_used = []
            
            key_libraries = [
                'tensorflow', 'keras', 'tf.keras', 'tf.data', 'tensorflow_hub',
                'pandas', 'numpy', 'matplotlib', 'sklearn', 'hub.keraslayer',
                'preprocessing layers', 'stringlookup', 'categoricalencoding'
            ]
            
            for lib in key_libraries:
                if lib in content_lower:
                    libraries_used.append(lib)
            
            if libraries_used:
                section += f"**Key Libraries Used:** {', '.join(libraries_used[:5])}\n\n"
            
            # Include COMPLETE tutorial content (no character limits)
            section += "**COMPLETE TUTORIAL CONTENT:**\n\n"
            section += "```markdown\n"
            section += tutorial.content
            section += "\n```\n\n"
            
            formatted_sections.append(section)
        
        # Add footer emphasizing adherence to tutorial patterns
        footer = """
**CRITICAL IMPLEMENTATION NOTES:**
1. **USE THE COMPLETE TUTORIAL CONTENT ABOVE as your primary reference**
2. Follow the exact import patterns, data loading, and preprocessing approaches shown
3. Adopt the same model architectures, training loops, and evaluation methods
4. Maintain consistency with the coding style and structure demonstrated
5. Prioritize the libraries and methods shown in tutorials over alternatives
6. **DO NOT SUMMARIZE OR ABBREVIATE - implement following the complete examples above**

"""
        formatted_sections.append(footer)
        
        return "\n".join(formatted_sections)
    
    def set_debug_mode(self, is_debug: bool = True) -> None:
        """Set debug mode state. When True, tutorials won't be included in prompts."""
        self.is_debug_mode = is_debug
        logger.info(f"Debug mode {'enabled' if is_debug else 'disabled'}")
    
    def reset_debug_mode(self) -> None:
        """Reset debug mode to False (normal mode)."""
        self.is_debug_mode = False
        logger.info("Debug mode reset - tutorials will be included in prompts")
    
    def _extract_all_code_blocks(self, content: str) -> List[str]:
        """Extract all code blocks from tutorial content."""
        code_blocks = []
        lines = content.split('\n')
        current_code_block = []
        in_code_block = False
        
        for line in lines:
            # Check for various code indicators
            line_stripped = line.strip()
            
            # Method 1: Traditional markdown code blocks
            if line_stripped.startswith('```') and ('python' in line_stripped or line_stripped == '```'):
                if in_code_block:
                    # End of code block
                    if current_code_block:
                        code_blocks.append('\n'.join(current_code_block).strip())
                        current_code_block = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
                continue
            
            # Method 2: Code blocks marked with "code:" prefix
            if line_stripped.startswith('code:') or line_stripped == 'code:':
                # Start collecting code until we hit a non-indented line or another section
                i = lines.index(line) + 1
                temp_code = []
                while i < len(lines):
                    next_line = lines[i]
                    # Stop if we hit another section marker or empty line followed by non-indented text
                    if (next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t') 
                        and not next_line.strip().startswith('#') and next_line.strip() != ''):
                        # Check if this looks like a section header
                        if any(keyword in next_line.lower() for keyword in ['step', 'section', 'part', 'tutorial', 'note:', 'example']):
                            break
                        # If it's a line that looks like regular code, include it
                        if any(char in next_line for char in ['=', '(', ')', '[', ']', 'import', 'def ', 'class ']):
                            temp_code.append(next_line)
                        else:
                            break
                    elif next_line.strip():  # Non-empty line
                        temp_code.append(next_line)
                    elif not next_line.strip() and temp_code:  # Empty line, but we have code
                        temp_code.append(next_line)
                    i += 1
                
                if temp_code:
                    # Clean up the code block
                    while temp_code and not temp_code[0].strip():
                        temp_code.pop(0)
                    while temp_code and not temp_code[-1].strip():
                        temp_code.pop()
                    if temp_code:
                        code_blocks.append('\n'.join(temp_code).strip())
                continue
            
            # Method 3: Lines that look like Python code (imports, functions, etc.)
            if not in_code_block and line_stripped:
                # Detect standalone code lines
                code_indicators = [
                    'import ', 'from ', 'def ', 'class ', 'if __name__',
                    '= tf.', '= pd.', '= np.', 'print(', '.fit(', '.predict(',
                    'model = ', 'dataset = ', 'data = '
                ]
                
                if any(indicator in line for indicator in code_indicators):
                    # Start collecting a code block
                    temp_code = [line]
                    line_idx = lines.index(line)
                    
                    # Look ahead for more code lines
                    i = line_idx + 1
                    while i < len(lines):
                        next_line = lines[i]
                        if not next_line.strip():  # Empty line
                            temp_code.append(next_line)
                        elif (next_line.startswith(' ') or next_line.startswith('\t') or
                              any(indicator in next_line for indicator in code_indicators) or
                              any(char in next_line for char in ['=', '(', ')', '[', ']', '.', '+', '-', '*', '/'])):
                            temp_code.append(next_line)
                        else:
                            break
                        i += 1
                    
                    if len(temp_code) > 1 or len(line.strip()) > 10:  # Only include substantial code blocks
                        # Clean up
                        while temp_code and not temp_code[-1].strip():
                            temp_code.pop()
                        if temp_code:
                            code_blocks.append('\n'.join(temp_code).strip())
            
            # Collect lines when in a markdown code block
            if in_code_block:
                current_code_block.append(line)
        
        # Handle any remaining code block
        if in_code_block and current_code_block:
            code_blocks.append('\n'.join(current_code_block).strip())
        
        # Remove duplicates and filter out very short snippets
        unique_blocks = []
        for block in code_blocks:
            if len(block.strip()) > 20 and block not in unique_blocks:  # Only substantial code blocks
                unique_blocks.append(block)
        
        return unique_blocks
