# projects/text_classification/ml_service/test_environment.py
def test_imports():
    """
    Test if all required packages are properly installed and functioning.
    This helps catch any environment setup issues early.
    """
    try:
        # Test basic imports
        import numpy as np
        print("✓ NumPy is installed correctly")
        
        import pandas as pd
        print("✓ Pandas is installed correctly")
        
        import sklearn
        print("✓ Scikit-learn is installed correctly")
        
        # Test basic functionality
        # If these operations work, our numerical computing setup is good
        arr = np.array([1, 2, 3])
        result = arr.mean()
        print("✓ NumPy operations working correctly")
        
        # If we get here, everything is working
        print("\nSuccess! Your environment is ready for machine learning!")
        print("\nCurrent versions:")
        print(f"NumPy: {np.__version__}")
        print(f"Pandas: {pd.__version__}")
        print(f"Scikit-learn: {sklearn.__version__}")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please check your conda environment and package installations")

if __name__ == "__main__":
    test_imports()
