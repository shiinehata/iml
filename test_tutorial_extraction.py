#!/usr/bin/env python3
"""
Test script to verify tutorial retriever modifications
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from iML.utils.tutorial_retriever import TutorialRetriever

def test_normal_mode():
    """Test normal mode - should return full tutorial content"""
    print("=" * 60)
    print("TESTING NORMAL MODE (should include full tutorials)")
    print("=" * 60)
    
    retriever = TutorialRetriever()
    
    # Mock analysis for image classification
    description_analysis = {
        'task_type': 'image classification',
        'task': 'binary classification',
        'input_data': 'image files',
    }
    
    tutorials = retriever.retrieve_tutorials(
        description_analysis=description_analysis,
        max_tutorials=1,
        is_debug_mode=False
    )
    
    print(f"Found {len(tutorials)} tutorials")
    if tutorials:
        formatted = retriever.format_tutorials_for_prompt(tutorials)
        print(f"Formatted prompt length: {len(formatted)} characters")
        print("First 500 characters of formatted prompt:")
        print(formatted[:500])
        print("...")
        print("Last 200 characters of formatted prompt:")
        print(formatted[-200:])

def test_debug_mode():
    """Test debug mode - should return empty tutorials"""
    print("\n" + "=" * 60)
    print("TESTING DEBUG MODE (should skip tutorials)")
    print("=" * 60)
    
    retriever = TutorialRetriever()
    
    # Mock analysis for image classification
    description_analysis = {
        'task_type': 'image classification',
        'task': 'binary classification',
        'input_data': 'image files',
    }
    
    tutorials = retriever.retrieve_tutorials(
        description_analysis=description_analysis,
        max_tutorials=1,
        is_debug_mode=True  # Debug mode enabled
    )
    
    print(f"Found {len(tutorials)} tutorials (should be 0)")
    
    # Test with some dummy tutorials to ensure format method also respects debug mode
    if not tutorials:
        # Create dummy tutorial to test format method in debug mode
        from iML.utils.tutorial_retriever import TutorialInfo
        from pathlib import Path
        dummy_tutorial = TutorialInfo(
            path=Path("dummy.md"),
            title="Dummy Tutorial",
            content="This is dummy content",
            score=1.0
        )
        formatted = retriever.format_tutorials_for_prompt([dummy_tutorial])
        print(f"Formatted prompt in debug mode: '{formatted}' (should be empty)")

def test_mode_switching():
    """Test switching between modes"""
    print("\n" + "=" * 60)
    print("TESTING MODE SWITCHING")
    print("=" * 60)
    
    retriever = TutorialRetriever()
    
    # Test setting debug mode manually
    retriever.set_debug_mode(True)
    print(f"Debug mode after set_debug_mode(True): {retriever.is_debug_mode}")
    
    retriever.reset_debug_mode()
    print(f"Debug mode after reset_debug_mode(): {retriever.is_debug_mode}")
    
    retriever.set_debug_mode(False)
    print(f"Debug mode after set_debug_mode(False): {retriever.is_debug_mode}")

if __name__ == "__main__":
    test_normal_mode()
    test_debug_mode()
    test_mode_switching()
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)